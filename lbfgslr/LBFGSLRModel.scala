package com.tencent.angel.ml.classification.lbfgslr

import com.tencent.angel.ml.classification.lr.SparseLRPredictResult
import com.tencent.angel.ml.conf.MLConf
import com.tencent.angel.ml.feature.LabeledData
import com.tencent.angel.ml.math.vector.DenseDoubleVector
import com.tencent.angel.ml.model.{MLModel, PSModel}
import com.tencent.angel.ml.predict.PredictResult
import com.tencent.angel.ml.utils.MathUtils
import com.tencent.angel.worker.storage.{DataBlock, MemoryDataBlock}
import com.tencent.angel.worker.task.TaskContext
import org.apache.commons.logging.LogFactory
import org.apache.hadoop.conf.Configuration

class LBFGSLRModel(conf: Configuration, _ctx: TaskContext = null) extends MLModel(conf, _ctx) {

    private val LOG = LogFactory.getLog(classOf[LBFGSLRModel])

    val m = conf.getInt("saveNum", 10) // save nums, a hyperparameter

    val LR_WEIGHT_MAT = "lr_weight"
    val LR_INTERCEPT = "lr_intercept"
    val LR_WEIGHT_GRAD = "lr_weight_grad"
    //val LR_INTERCEPT_GRAD = "lr_intercept_grad"
    val LR_WEIGHT_SAVE = "lr_weight_save"
    val LR_GRAD_SAVE = "lr_grad_save"
    val LR_DOT_SAVE = "lr_dot_matrix"
    val LR_LOSS = "lr_loss"
    val LR_DIRECTION = "lr_direction"
    val feaNum = conf.getInt(MLConf.ML_FEATURE_NUM, MLConf.DEFAULT_ML_FEATURE_NUM)

    // The feature weight vector, stored on PS
    val weight = PSModel[DenseDoubleVector](LR_WEIGHT_MAT, 1, feaNum).setAverage(false)
    val weight_grad = PSModel[DenseDoubleVector](LR_WEIGHT_GRAD, 1, feaNum).setAverage(true)
    val direction = PSModel[DenseDoubleVector](LR_DIRECTION, 1, feaNum).setAverage(false)
    val intercept_ = PSModel[DenseDoubleVector](LR_INTERCEPT, 1, 1).setAverage(true)

    val disW = PSModel[DenseDoubleVector](LR_WEIGHT_SAVE, m, feaNum).setAverage(false)
    val disG = PSModel[DenseDoubleVector](LR_GRAD_SAVE, m, feaNum).setAverage(false)
    // The first row of dosS is stale_grad, the other even row is disW and odd row is disG
    val dotS = PSModel[DenseDoubleVector](LR_DOT_SAVE, 2 * m + 1, feaNum, 2, 250000).setAverage(false)

    val totalLoss = PSModel[DenseDoubleVector](LR_LOSS, 1, 1).setAverage(true)

    val intercept =
      if (conf.getBoolean(MLConf.LR_USE_INTERCEPT, MLConf.DEFAULT_LR_USE_INTERCEPT)) {
        Some(intercept_)
      } else {
        None
      }

    addPSModel(LR_WEIGHT_MAT, weight)
    addPSModel(LR_INTERCEPT, intercept_)
    addPSModel(LR_WEIGHT_GRAD, weight_grad)
    addPSModel(LR_DIRECTION, direction)

    addPSModel(LR_WEIGHT_SAVE, disW)
    addPSModel(LR_GRAD_SAVE, disG)
    addPSModel(LR_DOT_SAVE, dotS)
    addPSModel(LR_LOSS, totalLoss)
    setSavePath(conf)
    setLoadPath(conf)


    /**
      *
      * @param dataSet
      * @return
      */
    override
    def predict(dataSet: DataBlock[LabeledData]): DataBlock[PredictResult] = {
      val start = System.currentTimeMillis()
      val wVector = weight.getRow(0)
      val b = intercept.map(_.getRow(0).get(0)).getOrElse(0.0)
      val cost = System.currentTimeMillis() - start
      LOG.info(s"pull LR Model from PS cost $cost ms." )

      val predict = new MemoryDataBlock[PredictResult](-1)

      dataSet.resetReadIndex()
      for (idx: Int <- 0 until dataSet.size) {
        val instance = dataSet.read
        val id = instance.getY
        val dot = wVector.dot(instance.getX)
        val sig = MathUtils.sigmoid(dot)
        predict.put(new SparseLRPredictResult(id, dot, sig))
      }
      predict
    }
}
