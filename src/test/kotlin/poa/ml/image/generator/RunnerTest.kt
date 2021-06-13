package poa.ml.image.generator

import org.junit.jupiter.api.Test

class RunnerTest {

    @Test
    internal fun testRunner() {
        val content = "/Users/oleg1024/Downloads/content_new"
        val style = "/Users/oleg1024/Downloads/style_new"
        val out = "/Users/oleg1024/Downloads/out_new"

        val alpha = "10"
        val betta = "0.00001,10"

        val styleWeights = "0.1,0.3,1.0,3.0,10.0;" +
                "0.2,0.2,0.2,0.2,0.2;" +
                "10.0,3.0,1.0,0.3,0.1"

        main(arrayOf(content, style, out, "15,50", alpha, betta, "0.03", styleWeights))

    }
}