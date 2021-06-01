package poa.ml.image.generator

import org.deeplearning4j.nn.api.Layer
import org.deeplearning4j.nn.api.ParamInitializer
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffOutputLayer
import org.deeplearning4j.nn.gradient.DefaultGradient
import org.deeplearning4j.nn.gradient.Gradient
import org.deeplearning4j.nn.layers.OutputLayer
import org.deeplearning4j.nn.params.EmptyParamInitializer
import org.deeplearning4j.nn.workspace.ArrayType
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr
import org.deeplearning4j.optimize.api.TrainingListener
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.common.primitives.Pair
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray

class CustomOutputLayer(
    conf: NeuralNetConfiguration,
    dataType: DataType,
    private val scoreFn: (INDArray, INDArray, LayerWorkspaceMgr) -> Double,
    private val gradFn: (INDArray, INDArray, LayerWorkspaceMgr) -> INDArray,
) : OutputLayer(conf, dataType) {

    private val defaultGradient = DefaultGradient()

    override fun computeScore(fullNetRegTerm: Double, training: Boolean, workspaceMgr: LayerWorkspaceMgr): Double =
        scoreFn(input, labels, workspaceMgr)

    override fun backpropGradient(epsilon: INDArray?, workspaceMgr: LayerWorkspaceMgr): Pair<Gradient, INDArray> =
        Pair.of(defaultGradient, gradFn(input, labels, workspaceMgr))

    override fun activate(training: Boolean, workspaceMgr: LayerWorkspaceMgr?): INDArray? {
        return workspaceMgr!!.createUninitialized(ArrayType.ACTIVATIONS, DataType.DOUBLE, *input.shape())
    }

}

class CustomOutputLayerConf(
    private val scoreFn: (SameDiff, SDVariable, SDVariable) -> SDVariable,
) : SameDiffOutputLayer() {

    override fun getOutputType(layerIndex: Int, inputType: InputType) = inputType

    override fun defineParameters(params: SDLayerParams) {
        //nothing
    }

    override fun initializeParameters(params: MutableMap<String, INDArray>) {
        //nothing
    }

    override fun defineLayer(
        sameDiff: SameDiff,
        layerInput: SDVariable,
        labels: SDVariable,
        paramTable: MutableMap<String, SDVariable>,
    ): SDVariable {
        return scoreFn(sameDiff, layerInput, labels)
    }

    override fun activationsVertexName() = "input"

}