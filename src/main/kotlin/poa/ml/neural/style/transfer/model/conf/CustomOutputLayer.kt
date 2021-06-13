package poa.ml.neural.style.transfer.model.conf

import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffOutputLayer
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.linalg.api.ndarray.INDArray

class SameDiffLambdaOutputLayerConf(
    private val scoreFn: (SameDiff, SDVariable, SDVariable) -> SDVariable,
) : SameDiffOutputLayer() {

    override fun getOutputType(layerIndex: Int, inputType: InputType) = inputType

    override fun defineParameters(params: SDLayerParams) {}

    override fun initializeParameters(params: MutableMap<String, INDArray>) {}

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