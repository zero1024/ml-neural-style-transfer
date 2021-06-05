package poa.ml.image.generator

import org.junit.jupiter.api.Test

class RunnerTest {

    @Test
    internal fun testRunner() {
        val content = "/Users/oleg1024/Downloads/content"
        val style = "/Users/oleg1024/Downloads/style"
        val out = "/Users/oleg1024/Downloads/out2"

        val alpha = "10"
        val betta = "40"

        main(arrayOf(content, style, out, "30", "0.1.2.3.5.15", alpha, betta))

    }
}