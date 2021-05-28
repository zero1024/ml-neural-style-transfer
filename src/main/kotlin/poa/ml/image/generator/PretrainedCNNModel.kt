package poa.ml.image.generator

import org.deeplearning4j.nn.conf.CNN2DFormat
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.graph.vertex.BaseWrapperVertex
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration
import org.deeplearning4j.nn.transferlearning.TransferLearning
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr
import org.nd4j.common.io.ClassPathResource
import org.nd4j.linalg.api.ndarray.INDArray

class PretrainedCNNModel(
    h5Path: String,
    width: Long,
    height: Long,
    nChannels: Long,
) {

    private val currentActivations = mutableMapOf<String, INDArray>()

    private val vgg19: ComputationGraph

    init {
        val modelHdf5Filename = ClassPathResource(h5Path).file.path
        val pretrained = KerasModelImport.importKerasModelAndWeights(modelHdf5Filename)

        vgg19 = TransferLearning.GraphBuilder(pretrained)
            .fineTuneConfiguration(FineTuneConfiguration())
            .setInputTypes(InputType.convolutional(
                height,
                width,
                nChannels,
                CNN2DFormat.NCHW
            ))
            .build()

        for (layer in vgg19.layers) {
            if (layer is org.deeplearning4j.nn.layers.normalization.BatchNormalization) {
                layer.layerConf().cnn2DFormat = CNN2DFormat.NCHW
            }
        }

        for ((idx, vertex) in vgg19.vertices.withIndex()) {
            vgg19.vertices[idx] = object : BaseWrapperVertex(vertex) {
                override fun doForward(training: Boolean, workspaceMgr: LayerWorkspaceMgr?): INDArray {
                    val res = super.doForward(training, workspaceMgr)
                    currentActivations[vertexName] = res
                    return res
                }
            }
        }
    }

    fun getActivations(img: INDArray, vararg layers: String): Map<String, INDArray> {
        vgg19.output(img)
        return currentActivations.filterKeys { layers.contains(it) }
    }
}
