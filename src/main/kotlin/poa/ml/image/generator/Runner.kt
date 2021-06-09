package poa.ml.image.generator

import org.bytedeco.opencv.opencv_core.Mat
import org.datavec.image.loader.NativeImageLoader
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.GradientUpdater
import org.nd4j.linalg.learning.config.Adam
import org.slf4j.LoggerFactory
import poa.ml.image.generator.model.DarknetNeuralTransferModel
import kotlin.math.min

private val logger = LoggerFactory.getLogger("main")

fun main(args: Array<String>) {

    val contentDir = args[0]
    val styleDir = args[1]
    val outDir = args[2]
    val iterations = args.getOrElse(3) { "50" }.toInt()
    val saveEvery = args.getOrElse(4) { "" }.split(".").map { it.toInt() }
    val alpha = args.getOrElse(5) { "10.0" }.toDouble()
    val betta = args.getOrElse(6) { "10.0" }.toDouble()
    val lr = args.getOrElse(7) { "0.03" }.toDouble()


    logger.info("Starting neural style transferring with parameters: ${args.toList()}")

    val imageLoader = NativeImageLoader()
    val model = DarknetNeuralTransferModel(alpha = alpha, betta = betta)

    logger.info("Loading styles...")
    val styles = mutableMapOf<String, INDArray>()
    walkFileTree(styleDir) { f ->
        val style = imageLoader.asMatrix(f)
        val resizedStyle = model.scaleForModel(style)
        styles[f.nameWithoutExtension] = resizedStyle
    }
    logger.info("Styles ${styles.keys} are loaded.")

    logger.info("Walking through content directory...")
    walkFileTree(contentDir) { f ->
        logger.info("Found [${f.name}]")
        val content = imageLoader.asMatrix(f)
        val origHeight = content.size(2).toInt()
        val origWidth = content.size(3).toInt()
        val (height, width) = chooseReasonableSize(origHeight, origWidth)
        val resizedContent = model.scaleForModel(content)

        for ((styleName, style) in styles) {
            logger.info("Applying style [${styleName}] to [${f.nameWithoutExtension}]")
            val label = model.toLabel(resizedContent, style)
            val updater = adamUpdater(lr, resizedContent.shape())

            var img = resizedContent.add(0.0)

            for (i in 0 until iterations) {
                val res = model.inputGradient(img, label)
                logger.info("Iteration $i. Score - ${model.score()}")
                updater.applyUpdater(res, i, 0)
                img = img.sub(res)
                if (saveEvery.contains(i) || i + 1 == iterations) {

                    val newImg = model.rescaleBack(img, width, height)
                    val outFilePath = "${outDir}/${f.nameWithoutExtension}_${styleName}_iter_${i}.jpg"
                    logger.info("Saving [${outFilePath}]...")
                    saveImage(outFilePath, imageLoader.asMat(newImg))
                    logger.info("Saving done.")
                }
            }
        }
    }
}

private fun adamUpdater(lr: Double, shape: LongArray): GradientUpdater<*> {
    return Adam(lr).instantiate(mapOf(
        "M" to Nd4j.create(*shape),
        "V" to Nd4j.create(*shape)),
        true)
}

private fun chooseReasonableSize(height: Int, width: Int): Pair<Int, Int> {
    val minSize = min(height, width)
    val reasonableMax = min(minSize, 500)
    val scale: Double = minSize.toDouble() / reasonableMax
    return (height.toDouble() / scale).toInt() to (width.toDouble() / scale).toInt()
}

private fun saveImage(path: String, img: Mat) {
    org.bytedeco.opencv.global.opencv_imgcodecs.imwrite(path, img)
}
