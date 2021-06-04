package poa.ml.image.generator

import org.datavec.image.loader.NativeImageLoader
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex.all
import org.nd4j.linalg.indexing.NDArrayIndex.point
import org.nd4j.linalg.indexing.conditions.GreaterThan
import org.nd4j.linalg.indexing.conditions.LessThan
import org.nd4j.linalg.learning.config.Adam
import org.slf4j.LoggerFactory
import poa.ml.image.generator.model.VGG19NeuralTransferModel
import java.awt.Color
import java.awt.image.BufferedImage
import java.awt.image.BufferedImage.TYPE_INT_RGB
import java.io.File
import javax.imageio.ImageIO
import kotlin.math.min

private val logger = LoggerFactory.getLogger("main")

fun main(args: Array<String>) {

    val contentDir = args[0]
    val styleDir = args[1]
    val outDir = args[2]
    val iterations = args.getOrElse(3) { "15" }.toInt()
    val alpha = args.getOrElse(4) { "10.0" }.toDouble()
    val betta = args.getOrElse(5) { "10.0" }.toDouble()
    val lr = args.getOrElse(6) { "0.03" }.toDouble()


    logger.info("Starting neural style transferring with parameters: ${args.toList()}")

    val imageLoader = NativeImageLoader()
    val vgg19 = VGG19NeuralTransferModel("vgg_19.h5", alpha = alpha, betta = betta)

    logger.info("Loading styles...")
    val styles = mutableMapOf<String, INDArray>()
    walkFileTree(styleDir) { f ->
        val style = imageLoader.asMatrix(f)
        scaleToZeroOne(style)
        val resizedStyle = imageLoader.resize(style, 400, 400)
        styles[f.nameWithoutExtension] = resizedStyle
    }
    logger.info("Styles ${styles.keys} are loaded.")

    logger.info("Walking through content directory...")
    walkFileTree(contentDir) { f ->
        logger.info("Found [${f.name}]")
        val content = imageLoader.asMatrix(f)
        scaleToZeroOne(content)
        val origHeight = content.size(2).toInt()
        val origWidth = content.size(3).toInt()
        val resizedContent = imageLoader.resize(content, 400, 400)

        for ((styleName, style) in styles) {
            logger.info("Applying style [${styleName}] to [${f.nameWithoutExtension}]")
            val label = vgg19.toLabel(resizedContent, style)
            val updater =
                Adam(lr).instantiate(mapOf(
                    "M" to Nd4j.create(*resizedContent.shape()),
                    "V" to Nd4j.create(*resizedContent.shape())),
                    true)

            var img = resizedContent.add(0.0)

            for (i in 0 until iterations) {
                logger.info("Iteration $i")
                val res = vgg19.inputGradient(img, label)
                updater.applyUpdater(res, i, 0)
                img = img.sub(res)
            }

            val (height, width) = chooseReasonableSize(origHeight, origWidth)
            val newImg = imageLoader.resize(scaleTo255(img), width, height)
            val outFilePath = "${outDir}/${f.nameWithoutExtension}-${styleName}.jpg"
            logger.info("Saving [${outFilePath}]...")
            saveImage(outFilePath, newImg)
            logger.info("Saving done.")
        }

    }
}

private fun chooseReasonableSize(height: Int, width: Int): Pair<Int, Int> {
    val minSize = min(height, width)
    val reasonableMax = min(minSize, 500)
    val scale: Double = minSize.toDouble() / reasonableMax
    return (height.toDouble() / scale).toInt() to (width.toDouble() / scale).toInt()
}

private fun scaleTo255(img: INDArray): INDArray {
    val newImg = img.add(0)
    newImg[all(), point(0), all(), all()].muli(255)
    newImg[all(), point(1), all(), all()].muli(255)
    newImg[all(), point(2), all(), all()].muli(255)
    newImg.replaceWhere(Nd4j.create(*img.shape()).assign(255), GreaterThan(255))
    newImg.replaceWhere(Nd4j.create(*img.shape()).assign(0), LessThan(0))
    return newImg
}

private fun scaleToZeroOne(styleImg: INDArray) {
    styleImg[all(), point(0), all(), all()].divi(255)
    styleImg[all(), point(1), all(), all()].divi(255)
    styleImg[all(), point(2), all(), all()].divi(255)
}

private fun saveImage(path: String, img: INDArray) {
    val height = img.size(2).toInt()
    val width = img.size(3).toInt()
    val bufImage = BufferedImage(width, height, TYPE_INT_RGB)

    for (w in 0 until width) {
        for (h in 0 until height) {
            val r = img.getScalar(0, 0, h, w).maxNumber().toInt()
            val g = img.getScalar(0, 1, h, w).maxNumber().toInt()
            val b = img.getScalar(0, 2, h, w).maxNumber().toInt()
            bufImage.setRGB(w, h, Color(b, g, r).rgb)
        }
    }

    ImageIO.write(bufImage, "jpg", File(path))
}
