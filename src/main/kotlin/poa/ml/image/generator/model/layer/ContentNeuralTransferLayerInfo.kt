package poa.ml.image.generator.model.layer

class ContentNeuralTransferLayerInfo(
    name: String,
    height: Long,
    width: Long,
    nChannels: Long,
) : NeuralTransferLayerInfo(name, height, width, nChannels
) {
    override fun flattenLayerName() = "${name}_flattened"
    override fun flattenLayerSize() = height * width * nChannels
}