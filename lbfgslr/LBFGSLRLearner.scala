package com.tencent.angel.ml.classification.lbfgslr

import com.tencent.angel.conf.AngelConf
import com.tencent.angel.ml.MLLearner
import com.tencent.angel.ml.conf.MLConf
import com.tencent.angel.ml.feature.LabeledData
import com.tencent.angel.ml.math.vector.{DenseDoubleVector, TDoubleVector}
import com.tencent.angel.ml.metric.log.LossMetric
import com.tencent.angel.ml.model.MLModel
import com.tencent.angel.ml.optimizer.sgd.{GradientDescent, L2LogLoss}
import com.tencent.angel.ml.optimizer.lbfgs.LBFGS
import com.tencent.angel.ml.utils.ValidationUtils
import com.tencent.angel.worker.storage.DataBlock
import com.tencent.angel.worker.task.TaskContext
import org.apache.commons.logging.LogFactory

class LBFGSLRLearner(ctx: TaskContext) extends MLLearner(ctx){
  private val LOG = LogFactory.getLog(classOf[LBFGSLRLearner])

  val epochNum: Int = conf.getInt(MLConf.ML_EPOCH_NUM, MLConf.DEFAULT_ML_EPOCH_NUM)
  val lr_0: Double = conf.getDouble(MLConf.ML_LEARN_RATE, MLConf.DEFAULT_ML_LEAR_RATE)
  val decay: Double = conf.getDouble(MLConf.ML_LEARN_DECAY, MLConf.DEFAULT_ML_LEARN_DECAY)
  val reg: Double = conf.getDouble(MLConf.ML_REG_L2, MLConf.DEFAULT_ML_REG_L2)
  val feaNum: Int = conf.getInt(MLConf.ML_FEATURE_NUM, MLConf.DEFAULT_ML_FEATURE_NUM)

  val m : Int= conf.getInt("saveNum", 10) // save nums, a hyperparameter
  val gamma : Double = conf.getDouble("gamma", 0.001) // origin search step, a hyperparameter
  val t: Double = conf.getDouble("ratioT", 2.0) // ratio of search step gain, a hyperparameter
  val epsilon: Double = conf.getDouble("epsilon", 0.0001) // threshold for linear search, a hyperparamter

  // Init Model
  val lrModel = new LBFGSLRModel(conf, ctx)
  // LR uses log loss
  val l2LL = new L2LogLoss(reg)

  def trainOneEpoch(epoch: Int, trainData: DataBlock[LabeledData]): DenseDoubleVector = {

    // Decay learning rate.
    val lr = lr_0 / Math.sqrt(1.0 + decay * epoch)
    //val lr = lr_0
    val mygamma = gamma / Math.sqrt(1.0 + decay * epoch)
    val taskId = ctx.getTaskIndex
    // Apply LBFGS
    val startBatch = System.currentTimeMillis()
    val (totalLoss, localweight) = LBFGS.runLBFGS(trainData, lrModel, l2LL, epoch, m, gamma, lr, t, epsilon, taskId, ctx)
    val batchCost = System.currentTimeMillis() - startBatch

    if(taskId == 0){
      LOG.info(s"Task[${ctx.getTaskIndex}]: epoch=$epoch LBFGS update success." +
        s"Cost $batchCost ms. " +
        s"Total loss = $totalLoss")
    }
    localweight
  }

  override def train(trainData: DataBlock[LabeledData], validationData: DataBlock[LabeledData]): MLModel = {

    LOG.info(s"Task[${ctx.getTaskIndex}]: Starting to train a LR model...")
    LOG.info(s"Task[${ctx.getTaskIndex}]: epoch=$epochNum, initLearnRate=$lr_0, " + s"learnRateDecay=$decay, L2Reg=$reg"
    + s" gamma = $gamma, epsilon = $epsilon")

    globalMetrics.addMetrics(MLConf.TRAIN_LOSS, LossMetric(trainData.size))
    //LOG.info(s"here")
    globalMetrics.addMetrics(MLConf.VALID_LOSS, LossMetric(validationData.size))
    //LOG.info(s"here")
    while (ctx.getEpoch < epochNum) {
      val epoch = ctx.getEpoch
      LOG.info(s"Task[${ctx.getTaskIndex}]: epoch=$epoch start.")

      val startTrain = System.currentTimeMillis()
      val localWeight = trainOneEpoch(epoch, trainData)
      val trainCost = System.currentTimeMillis() - startTrain

      val startValid = System.currentTimeMillis()
      validate(epoch, localWeight, trainData, validationData)
      val validCost = System.currentTimeMillis() - startValid

      LOG.info(s"Task[${ctx.getTaskIndex}]: epoch=$epoch success. " +
        s"epoch cost ${trainCost + validCost} ms." +
        s"train cost $trainCost ms. " +
        s"validation cost $validCost ms.")
      LOG.info(s"Task[${ctx.getTaskIndex}]: epoch=$epoch success. " +
              s"train cost $trainCost ms. ")
      ctx.incEpoch()
    }

    lrModel
  }

  /**
    * validate loss, Auc, Precision or other
    *
    * @param epoch          : epoch id
    * @param valiData : validata data storage
    */
  def validate(epoch: Int, weight: TDoubleVector, trainData: DataBlock[LabeledData], valiData: DataBlock[LabeledData]) = {
    val trainMetrics = ValidationUtils.calMetrics(trainData, weight, l2LL)
    LOG.info(s"Task[${ctx.getTaskIndex}]: epoch = $epoch " +
      s"trainData loss = ${trainMetrics._1 / trainData.size()} " +
      s"precision = ${trainMetrics._2} " +
      s"auc = ${trainMetrics._3} " +
      s"trueRecall = ${trainMetrics._4} " +
      s"falseRecall = ${trainMetrics._5}")
    globalMetrics.metrics(MLConf.TRAIN_LOSS, trainMetrics._1)

    if (valiData.size > 0) {
      val validMetric = ValidationUtils.calMetrics(valiData, weight, l2LL);
      LOG.info(s"Task[${ctx.getTaskIndex}]: epoch=$epoch " +
        s"validationData loss=${validMetric._1 / valiData.size()} " +
        s"precision=${validMetric._2} " +
        s"auc=${validMetric._3} " +
        s"trueRecall=${validMetric._4} " +
        s"falseRecall=${validMetric._5}")
      globalMetrics.metrics(MLConf.VALID_LOSS, validMetric._1)
    }
  }
}


