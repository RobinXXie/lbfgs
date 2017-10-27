package com.tencent.angel.ml.classification.lbfgslr

import com.tencent.angel.ml.MLRunner
import com.tencent.angel.ml.classification.lr.{LRModel, LRPredictTask, LRRunner, LRTrainTask}
import org.apache.commons.logging.LogFactory
import org.apache.hadoop.conf.Configuration

class LBFGSLRRunner extends MLRunner{
  private val LOG = LogFactory.getLog(classOf[LBFGSLRRunner])
  override
  def train(conf: Configuration): Unit = {
    conf.setInt("angel.worker.matrixtransfer.request.timeout.ms", 60000)

    train(conf, new LBFGSLRModel(conf), classOf[LBFGSLRTrainTask])
  }
  override
  def predict(conf: Configuration): Unit = {
    conf.setInt("angel.worker.matrix.transfer.request.timeout.ms", 60000)
    predict(conf, new LBFGSLRModel(conf), classOf[LBFGSLRPredictTask])
  }
  override
  def incTrain(conf: Configuration): Unit = ???
}
