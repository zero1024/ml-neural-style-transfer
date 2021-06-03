package poa.ml.image.generator

import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaLayer
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffOutputLayer
import org.nd4j.autodiff.samediff.SDIndex.point
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.linalg.api.ndarray.INDArray

class CustomLayerConf(private val mb: Long = 1) : SameDiffLambdaLayer() {

    override fun defineLayer(sameDiff: SameDiff, input: SDVariable) = sameDiff.convGramMatrix(mb, input)

    override fun getOutputType(layerIndex: Int, inputType: InputType): InputType {
        inputType as InputType.InputTypeConvolutional
        return InputType.feedForward(inputType.channels * inputType.channels)
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