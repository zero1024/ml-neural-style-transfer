package poa.ml.neural.style.transfer.model.layer

class StyleNeuralTransferLayerInfo(
    name: String,
    height: Long,
    width: Long,
    nChannels: Long,
    val weight: Double = 0.2,
) : NeuralTransferLayerInfo(name, height, width, nChannels
) {
    override fun flattenLayerName() = "${name}_flattened_gram"
    override fun flattenLayerSize() = nChannels * nChannels
}