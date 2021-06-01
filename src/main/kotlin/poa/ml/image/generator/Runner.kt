package poa.ml.image.generator

import org.apache.commons.lang3.SerializationUtils
import org.datavec.image.loader.NativeImageLoader
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.rng.CpuNativeRandom
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex.*
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4j.rng.NativeRandom
import java.awt.Color
import java.awt.Dimension
import java.awt.FlowLayout
import java.awt.Toolkit
import java.awt.image.BufferedImage
import java.awt.image.BufferedImage.TYPE_INT_RGB
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import javax.imageio.ImageIO
import javax.swing.ImageIcon
import javax.swing.JFrame
import javax.swing.JLabel
import javax.swing.WindowConstants

class Runner {

}

fun main() {
    val imageLoader = NativeImageLoader(400, 400, 3)


    val vgg19 = VGG19NeuralTransferModel("vgg_19.h5", 400, 400, 3)

    val contentImage = imageLoader.asMatrix(File("/Users/oleg1024/Downloads/1/photo_2021-05-28_17-36-39.jpg"))

    contentImage[all(), point(0), all(), all()].divi(contentImage[all(), point(0), all(), all()].maxNumber())
    contentImage[all(), point(1), all(), all()].divi(contentImage[all(), point(1), all(), all()].maxNumber())
    contentImage[all(), point(2), all(), all()].divi(contentImage[all(), point(2), all(), all()].maxNumber())

    var img = contentImage.add(0.0)

    val rand = Nd4j.rand(0.1, 0.9, CpuNativeRandom(), *contentImage.shape())
    img.addi(rand)
    img[all(), point(0), all(), all()].divi(img[all(), point(0), all(), all()].maxNumber())
    img[all(), point(1), all(), all()].divi(img[all(), point(1), all(), all()].maxNumber())
    img[all(), point(2), all(), all()].divi(img[all(), point(2), all(), all()].maxNumber())

    //0.01719984130859375
//    var img = Nd4j.rand(*contentImage.shape())
//    img[all(), point(0), all(), all()].divi(img[all(), point(0), all(), all()].maxNumber())
//    img[all(), point(1), all(), all()].divi(img[all(), point(1), all(), all()].maxNumber())
//    img[all(), point(2), all(), all()].divi(img[all(), point(2), all(), all()].maxNumber())

//    var img =
//        FileInputStream(File("/Users/oleg1024/Downloads/some")).use { SerializationUtils.deserialize(it) as INDArray }


    val label = vgg19.feedForward(contentImage)["block5_conv4"]!!
    val label2D = Nd4j.create(label.size(0), vgg19.getLabelSize())
    label2D[all(), interval(label2D.size(1) - vgg19.contentLayer.flattenSize(), label2D.size(1))]
        .assign(labels2d(label))

    val updater =
        Adam(0.03).instantiate(mapOf("M" to Nd4j.create(*img.shape()), "V" to Nd4j.create(*img.shape())), true)

    while (true) {

        val newImg = img.add(0)
        newImg[all(), point(0), all(), all()].muli(103.939 * 2)
        newImg[all(), point(1), all(), all()].muli(116.779 * 2)
        newImg[all(), point(2), all(), all()].muli(123.68 * 2)
        showImage(Transforms.relu(newImg).mul(newImg.lt(256).castTo(DataType.DOUBLE)))

        for (i in 0 until 10) {
            val res = vgg19.getInputGradient(img, label2D)
//            res.negi()
            updater.applyUpdater(res, i, 0)
            img = img.sub(updater.state["M"]!!)
        }
        FileOutputStream(File("/Users/oleg1024/Downloads/some")).use { SerializationUtils.serialize(img, it) }
    }

}

private fun labels2d(labels: INDArray): INDArray {
    val mb = labels.size(0)
    val labelsAsList = labels.shape().toList()
    return labels.reshape(mb, labelsAsList.subList(0, labelsAsList.size).reduce { i1, i2 -> i1 * i2 })
}

private fun showImage(img: INDArray) {
    val bufImage = BufferedImage(400, 400, TYPE_INT_RGB)

    for (w in 0 until 400) {
        for (h in 0 until 400) {
            val r = img.getScalar(0, 0, h, w).maxNumber().toInt()
            val g = img.getScalar(0, 1, h, w).maxNumber().toInt()
            val b = img.getScalar(0, 2, h, w).maxNumber().toInt()
            bufImage.setRGB(w, h, Color(b, g, r).rgb)
        }
    }
    ImageIO.write(bufImage, "jpg", File("/Users/oleg1024/Downloads/some.jpg"))
    showImage(bufImage)
}

fun showImage(image: BufferedImage, lambda: (JFrame) -> Unit = {}) {
    showJLabel(JLabel(ImageIcon(image)), lambda)
}

fun showJLabel(jLabel: JLabel, lambda: (JFrame) -> Unit = {}) {
    val frame = JFrame()
    val dim: Dimension = Toolkit.getDefaultToolkit().screenSize
    frame.contentPane.layout = FlowLayout()
    frame.contentPane.add(jLabel)
    frame.defaultCloseOperation = WindowConstants.DISPOSE_ON_CLOSE
    lambda(frame)
    frame.pack()
    frame.setLocation(dim.width / 2 - frame.size.width / 2, dim.height / 2 - frame.size.height / 2)
    frame.isVisible = true
}