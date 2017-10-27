package com.tencent.angel.ml.optimizer.lbfgs

import com.tencent.angel.exception.AngelException
import com.tencent.angel.ml.classification.lbfgslr.LBFGSLRModel
import com.tencent.angel.ml.feature.LabeledData
import com.tencent.angel.ml.math.TAbstractVector
import com.tencent.angel.ml.math.vector.{DenseDoubleVector, SparseDoubleVector, TDoubleVector}
import com.tencent.angel.ml.matrix.psf.aggr.Dot
import com.tencent.angel.ml.matrix.psf.aggr.enhance.ScalarAggrResult
import com.tencent.angel.ml.matrix.psf.get.base.GetFunc
import com.tencent.angel.ml.matrix.psf.update.{Fill, RandomNormal}
import com.tencent.angel.ml.matrix.psf.update.enhance.UpdateFunc
import com.tencent.angel.ml.metric.log.GlobalMetrics
import com.tencent.angel.ml.model.PSModel
import com.tencent.angel.ml.optimizer.sgd.Loss
import com.tencent.angel.worker.storage.DataBlock
import com.tencent.angel.worker.task.TaskContext
import org.apache.commons.logging.{Log, LogFactory}
import com.tencent.angel.ml.matrix.psf.aggr.enhance.ScalarAggrResult

import scala.math.abs

//In this algorithm weight_grad is non_neg, H^-1g is non_neg
object LBFGS {
  private val LOG: Log = LogFactory.getLog(LBFGS.getClass)

  def runLBFGS(data: DataBlock[LabeledData],
               model: LBFGSLRModel,
               loss: Loss,
               epoch: Int,
               m: Int,
               gamma: Double,
               lr: Double,
               t: Double,
               epsilon: Double,
               taskId: Int,
               ctx: TaskContext): (Double, DenseDoubleVector) = {
    val startsyn = System.currentTimeMillis()

    //initialize the model weight
//    if (epoch == 0){
//      //LOG.info(s"Training_data size is ${data.size()}")
//      if(taskId == 0){
//        model.weight.update(new RandomNormal(model.weight.getMatrixId(), 0, 0, 0.01)).get()
//        model.weight.syncClock()
//      }
//      else{
//        model.weight.syncClock(false)
//      }
//    }

    //get weight and bias and stalegrad
    val w = model.weight.getRow(0)
    var b: Option[Double] = model.intercept.map(_.getRow(0).get(0))
    val stale_grad = model.weight_grad.getRow(0)

    // zero the parameter on ps
    model.totalLoss.syncClock(false)
    model.totalLoss.getRow(0)
//    model.weight_grad.syncClock(false)
//    model.weight_grad.getRow(0)
    if(taskId == 0) {
      model.totalLoss.zero()
      model.weight_grad.zero()
    }
    model.totalLoss.syncClock(false)
    model.totalLoss.getRow(0)
//    model.weight_grad.syncClock(false)
//    model.weight_grad.getRow(0)


    //wait for synchronizing
//    model.totalLoss.getRow(0)
//    model.weight_grad.getRow(0)
    val endsyn = System.currentTimeMillis()
    LOG.info(s"Task[${ctx.getTaskIndex}]: epoch=$epoch" +
      s"Time of initialization synchronzing = ${endsyn - startsyn} ms")


    //calculate gradient
    val batchloss = calgrad(data, model, loss, epoch, lr, w, b)
    LOG.info(s"Task[${ctx.getTaskIndex}]: epoch=$epoch" +
      s"Batch loss is  = $batchloss")
    val endgrad = System.currentTimeMillis()

    //get gradient and temporary loss
    val grad = model.weight_grad.getRow(0)
    val newb: Option[Double] = model.intercept.map(_.getRow(0).get(0))
    val totalLoss = model.totalLoss.getRow(0).get(0)

    if (epoch != 0){
      myUpdate2(model, taskId, epoch, m, stale_grad, grad)
    }
    //calculate direction has some problem
    caldire(taskId, epoch, m, model, w, grad)
    val enddire = System.currentTimeMillis()
    LOG.info(s"Task[${ctx.getTaskIndex}]: epoch=$epoch" +
      s"Time of caculate diretion = ${enddire - endgrad} ms")

    //get direction
    val dire = model.direction.getRow(0)

    //calculate area of step
    val stepArea = forwardAndBack(data, model, gamma, t, taskId, loss, epoch, w, dire, totalLoss, newb, epsilon)
    val endarea = System.currentTimeMillis()
    val startStep = stepArea._1
    val endStep = stepArea._2
    LOG.info(s"Task[${ctx.getTaskIndex}]: epoch=$epoch" +
      s"The area of search is [$startStep, $endStep]" +
      s"Time of caculate area = ${endarea - enddire} ms")

    val finalStep = calstep(data, model, loss, taskId, startStep, endStep, epsilon, epoch, w, newb, dire)
    val endstep = System.currentTimeMillis()
    LOG.info(s"Task[${ctx.getTaskIndex}]: epoch=$epoch" +
      s"Time of caculate step = ${endstep - endarea} ms" +
      s"Final step is $finalStep")


    myUpdate(model, finalStep, taskId, dire, epoch, m, grad)
    val endupdate = System.currentTimeMillis()
    LOG.info(s"Task[${ctx.getTaskIndex}]: epoch=$epoch" +
      s"Time of caculate update = ${endupdate - endstep} ms")

    (totalLoss, w)
  }

  //This function changed weight_grad, intecept and totalloss
  def calgrad[M <: TDoubleVector](trainData: DataBlock[LabeledData],
                                  model: LBFGSLRModel,
                                  loss: Loss,
                                  epoch: Int,
                                  lr: Double,
                                  w: DenseDoubleVector,
                                  b: Option[Double]) : Double = {
    //val batchStartTs = System.currentTimeMillis()
    val grad = new DenseDoubleVector(w.getDimension)
    grad.setRowId(0)

    val bUpdate = new DenseDoubleVector(1)
    bUpdate.setRowId(0)

    var batchLoss: Double = 0.0
    var gradScalarSum: Double = 0.0

    for (i <- 0 until trainData.size()) {
      val (x: TAbstractVector, y: Double) = loopingData(trainData)
      val pre = w.dot(x) + b.getOrElse(0.0)
      val gradScalar = -loss.grad(pre, y)    // not negative gradient
      grad.plusBy(x, gradScalar)
      batchLoss += loss.loss(pre, y)
      gradScalarSum += gradScalar
    }

    grad.timesBy(1.toDouble / trainData.size().asInstanceOf[Double])
    gradScalarSum /= trainData.size()

    if (loss.isL2Reg) {
      for (index <- 0 until grad.size) {
        if (grad.get(index) > 10e-7) {
          grad.set(index, grad.get(index) + w.get(index) * (loss.getRegParam))
        }
      }
    }
    if (loss.isL1Reg) {
      truncGradient(grad, 0, loss.getRegParam)//?
    }

    model.weight_grad.increment(grad.timesBy(1.0).asInstanceOf[M])
    model.intercept.map { bv =>
      bUpdate.set(0, -lr * gradScalarSum)
      bv.increment(bUpdate)
      bv
    }
    //LOG.debug(s"Epoch[$epoch] Local loss = $batchLoss")
    //taskContext.updateProfileCounter(trainData.size(), (System.currentTimeMillis() - batchStartTs).toInt)//?
    batchLoss /= trainData.size()
    batchLoss += loss.getReg(w)

    val deltaLoss = new DenseDoubleVector(1)
    deltaLoss.setRowId(0)
    deltaLoss.set(0, batchLoss)
    model.totalLoss.increment(deltaLoss)

    //Push model update to PS Server
    model.weight_grad.syncClock()
    model.intercept.map(_.syncClock())
    model.totalLoss.syncClock()

    batchLoss
  }

  //This function has changed the direction
  def caldire[M <: TDoubleVector](id : Int,
                                  epoch: Int,
                                  m: Int,
                                  model: LBFGSLRModel,
                                  w: DenseDoubleVector,
                                  grad: DenseDoubleVector): Unit ={
    model.direction.syncClock(false)
    model.direction.getRow(0)
    if(id == 0){
      //val w = model.weight.getRow(0)
      //val grad = model.weight_grad.getRow(0)
      val dire = new DenseDoubleVector(w.getDimension)
      dire.setRowId(0)
      dire.clone(grad)


      val start = if(epoch <= m) 0 else epoch - m
      val saveNum = if (epoch <= m) epoch else m
      //val saveDot = model.dotS.getRow(0)
      var index = saveNum - 1
      val alpha = new DenseDoubleVector(m)
      alpha.setRowId(0)

      val dotArr = new Array[Double](m)
      while(index >= 0) {
        val realindex = index + start
        val saveindex = realindex % m
        val disGindex = 2 * (saveindex + 1) - 1
        val windex = model.dotS.getRow(disGindex - 1)
        val gindex = model.dotS.getRow(disGindex)

        val result = model.dotS.get(new Dot(model.dotS.getMatrixId, disGindex - 1, disGindex)).asInstanceOf[ScalarAggrResult].getResult
        dotArr(saveindex) = result
        val p = 1 / result
        val sq = windex.dot(dire)
        val newalpha = 1.0 * p * sq
        val alphaY = gindex.times(-1.0 * newalpha)

        dire.plusBy(alphaY)
        alpha.set(index, newalpha)
        index = index - 1
      }
      for(index <- 0 until saveNum){
        val realindex = index + start
        val saveindex = realindex % m
        val disGindex = 2 * (saveindex + 1) - 1
        val windex = model.dotS.getRow(disGindex - 1)
        val gindex = model.dotS.getRow(disGindex)
        val p = 1 / (dotArr(saveindex))
        val yq = gindex.dot(dire)
        val beta = 1.0 * p * yq
        val betaS = windex.times(alpha.get(index) - beta)

        dire.plusBy(betaS)
        //dire.timesBy(-1.0)
      }
      model.direction.zero()
      model.direction.increment(dire)
      model.direction.syncClock()
    }
    else{
      model.direction.clock(false).get
    }
  }

  //this function changes totoalloss
  def forwardAndBack(trainData: DataBlock[LabeledData],
                     model: LBFGSLRModel,
                     gamma: Double,
                     t: Double,
                     taskId: Int,
                     loss: Loss,
                     epoch: Int,
                     newW: DenseDoubleVector,
                     dire: DenseDoubleVector,
                     totalLoss: Double,
                     b: Option[Double],
                     epsilon: Double): (Double, Double) ={
    //val newW = model.weight.getRow(0)
    //val dire = model.direction.getRow(0)
    //val totalLoss = model.totalLoss.getRow(0).get(0)
    //var b: Option[Double] = model.intercept.map(_.getRow(0).get(0))

    var alpha0: Double = 0.0
    var alpha1: Double = 0.0
    var alpha: Double = 0.0

    var step : Double = gamma

    var startStep : Double = 0.0
    var endStep: Double = 0.0

    var alpha0Loss : Double = totalLoss
    var alpha1Loss : Double = 0.0
    var flag = 0
    var times = 0
    while(flag == 0){
      times = times + 1
      alpha1 = alpha0 + step
      newW.plusBy(dire.times(-1.0 * alpha1))

      //caculate loss at a new point newW
      //flush stale loss
      model.totalLoss.syncClock(false)
      model.totalLoss.getRow(0)
      if(taskId == 0){
        model.totalLoss.zero()
      }
      model.totalLoss.syncClock(false)
      model.totalLoss.getRow(0)

      //caculate new loss
      var batchLoss : Double= 0.0
      for (i <- 0 until trainData.size()) {
        val (x: TAbstractVector, y: Double) = loopingData(trainData)
        val pre = newW.dot(x) + b.getOrElse(0.0)
        batchLoss += loss.loss(pre, y)
      }
      batchLoss /= trainData.size()
      batchLoss += loss.getReg(newW)

      //LOG.info(s"Task[${taskId}]: batch_alpha1_Loss is $batchLoss")

      val deltaLoss = new DenseDoubleVector(1)
      deltaLoss.setRowId(0)
      deltaLoss.set(0, batchLoss)
     // LOG.info(s"Task[${taskId}]: delta_alpha1_Loss is ${deltaLoss.get(0)}")
      model.totalLoss.increment(deltaLoss)

      //Push batchloss  to PS Server
      model.totalLoss.syncClock()

      //get newW loss
      alpha1Loss= model.totalLoss.getRow(0).get(0)
      LOG.info(s"alpha1 is $alpha1, and alpha1 loss is $alpha1Loss")
      if(alpha1Loss > alpha0Loss || abs(alpha1Loss - alpha0Loss) < epsilon * 0.1){
        startStep = math.min(alpha, alpha1)
        endStep = math.max(alpha, alpha1)
        newW.plusBy(dire.times(alpha1))
        flag = 1
      }
      else{
        step = step * t
        alpha = alpha0
        alpha0 = alpha1
        newW.plusBy(dire.times(alpha1))
        alpha0Loss = alpha1Loss
      }
    }
    LOG.info(s"It find $times steps to find the sup and inf")
    (startStep, endStep)
  }
  def calstep(trainData : DataBlock[LabeledData],
              model: LBFGSLRModel,
              loss : Loss,
              taskId: Int,
              startStep: Double,
              endStep: Double,
              epsilon: Double,
              epoch: Int,
              w: DenseDoubleVector,
              newb: Option[Double],
              direction: DenseDoubleVector): Double ={
    var a = startStep
    var b = endStep
    var alLoss : Double = 0
    var arLoss : Double = 3 * epsilon
    var times = 0
    while(abs(alLoss - arLoss) > epsilon){
      times = times + 1
      val al = a + 0.382 * (b - a)
      val ar = a + 0.618 * (b - a)
      alLoss = calLoss(trainData, model, loss, al, taskId, w, newb, direction)
      arLoss = calLoss(trainData, model, loss, ar, taskId, w, newb, direction)
      LOG.info(s"al and ar is $al and $ar. The loss of al and ar is $alLoss and $arLoss")
      if (alLoss < arLoss){
        b = ar
      }
      else{
        a = al
      }
    }
    LOG.info(s"The iteration in finding step is $times")
    (a + b) / 2
  }
  def calLoss(trainData: DataBlock[LabeledData],
              model: LBFGSLRModel,
              loss: Loss,
              step: Double,
              taskId: Int,
              newW: DenseDoubleVector,
              b: Option[Double],
              dire: DenseDoubleVector): Double = {
    //val newW = model.weight.getRow(0)
    //val b : Option[Double] = model.intercept.map(_.getRow(0).get(0))
    //val dire = model.direction.getRow(0)
    newW.plusBy(dire.times(-1.0 * step))

    //caculate loss at a new point newW
    //flush stale loss
    model.totalLoss.syncClock(false)
    model.totalLoss.getRow(0)
    if(taskId == 0){
      model.totalLoss.zero()
    }
    model.totalLoss.syncClock(false)
    model.totalLoss.getRow(0)

    //caculate new loss
    var batchLoss : Double= 0.0
    for (i <- 0 until trainData.size()) {
      val (x: TAbstractVector, y: Double) = loopingData(trainData)
      val pre = newW.dot(x) + b.getOrElse(0.0)
      batchLoss += loss.loss(pre, y)
    }
    batchLoss /= trainData.size()
    batchLoss += loss.getReg(newW)


    val deltaLoss = new DenseDoubleVector(1)
    deltaLoss.setRowId(0)
    deltaLoss.set(0, batchLoss)

    newW.plusBy(dire.times(step))
    model.totalLoss.increment(deltaLoss)
    //Push batchloss  to PS Server
    model.totalLoss.syncClock()
    //get newW loss
    model.totalLoss.getRow(0).get(0)
  }
  def myUpdate(model: LBFGSLRModel, step: Double, taskId : Int, dire : DenseDoubleVector,
               epoch : Int,
               m: Int,
               grad: DenseDoubleVector): Unit ={
//    model.weight_grad.syncClock(false)
//    model.weight_grad.getRow(0)
    val disWrow = (epoch % m) * 2 + 1 - 1
    model.dotS.syncClock(false)
    model.dotS.getRow(0)
    if(taskId == 0){
      val delta = dire.times(-1.0 * step)
      delta.setRowId(0)
      //val row = epoch % m
      model.weight.increment(delta)
      model.weight.syncClock()

      model.dotS.update(new Fill(model.dotS.getMatrixId, disWrow, 0)).get()
      model.dotS.increment(disWrow, delta)
      model.dotS.syncClock()

    }
    else{
      model.weight.clock(false).get
      model.dotS.syncClock(false)
    }
    model.dotS.getRow(0)
  }
  def myUpdate2(model: LBFGSLRModel, taskId : Int,
                epoch : Int,
                m: Int,
                stalegrad: DenseDoubleVector,
                grad: DenseDoubleVector): Unit ={
    val row = 2 * ((epoch - 1) % m + 1) - 1
    model.dotS.syncClock(false)
    model.dotS.getRow(row)
    if(taskId == 0){
      model.dotS.update(new Fill(model.dotS.getMatrixId, row, 0)).get()
      model.dotS.increment(row, grad.plus(stalegrad, -1.0))
      model.dotS.syncClock()
    }
    else{
      model.dotS.syncClock(false)
    }
    model.dotS.getRow(row)

  }
  def loopingData(trainData: DataBlock[LabeledData]): (TAbstractVector, Double) = {
    var data = trainData.read()
    if (data == null) {
      trainData.resetReadIndex()
      data = trainData.read()
    }

    if (data != null)
      (data.getX, data.getY)
    else
      throw new AngelException("Train data storage is empty or corrupted.")
  }
  def truncGradient(vec: TDoubleVector, alpha: Double, theta: Double): TDoubleVector = {
    val update = new SparseDoubleVector(vec.getDimension)

    for (dim <- 0 until update.getDimension) {
      val value = vec.get(dim)
      if (value >= 0 && value <= theta) {
        val newValue = if (value - alpha > 0) value - alpha else 0
        vec.set(dim, newValue)
        update.set(dim, newValue - value)
      } else if (value < 0 && value >= -theta) {
        val newValue = if (value - alpha < 0) value - alpha else 0
        vec.set(dim, newValue)
        update.set(dim, newValue - value)
      }
    }

    update
  }
}
