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

private val logger = LoggerFactory.getLogger("root")

fun main(args: Array<String>) {

    val contentDir = args[0]
    val styleDir = args[1]
    val outDir = args[2]
    val iterations = args.getOrElse(3) { "15" }.toInt()

    val imageLoader = NativeImageLoader()
    val vgg19 = VGG19NeuralTransferModel("vgg_19.h5")

    val styles = mutableMapOf<String, INDArray>()
    walkFileTree(styleDir) { f ->
        val style = imageLoader.asMatrix(f)
        scaleToZeroOne(style)
        val resizedStyle = imageLoader.resize(style, 400, 400)
        styles[f.nameWithoutExtension] = resizedStyle
    }

    walkFileTree(contentDir) { f ->
        val content = imageLoader.asMatrix(f)
        scaleToZeroOne(content)
        val height = content.size(2).toInt()
        val width = content.size(3).toInt()
        val resizedContent = imageLoader.resize(content, 400, 400)

        for ((styleName, style) in styles) {
            val label = vgg19.toLabel(resizedContent, style)
            val updater =
                Adam(0.03).instantiate(mapOf(
                    "M" to Nd4j.create(*resizedContent.shape()),
                    "V" to Nd4j.create(*resizedContent.shape())),
                    true)

            var img = resizedContent.add(0.0)

            for (i in 0 until iterations) {
                val res = vgg19.inputGradient(img, label)
                updater.applyUpdater(res, i, 0)
                img = img.sub(res)
            }

            val newImg = imageLoader.resize(scaleTo255(img), width, height)
            saveImage("${outDir}/${f.nameWithoutExtension}-${styleName}.jpg", newImg)


        }

    }
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
