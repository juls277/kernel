import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import java.awt.Color;
import java.awt.image.BufferedImage;
public class Parallel {
    public static int HEIGHT;
    public static int WIDTH;

    public static BufferedImage input,output;


        public static void convolutionImage(String filepath, int order, float factor, float bias, float[][] kernel) throws IOException {
            input = ImageIO.read(new File(filepath));

            WIDTH = input.getWidth();
            HEIGHT = input.getHeight();




            output = new BufferedImage(WIDTH, HEIGHT, input.getType());
            float mult_factor = factor;

            // Number of threads
            int numThreads = Runtime.getRuntime().availableProcessors();

            // Create an array to hold threads
            Thread[] threads = new Thread[WIDTH];

            // Divide the work among threads
            int chunkSize = WIDTH / numThreads;
            for (int i = 0; i < numThreads; i++) {
                int start = i * chunkSize;
                int end = (i == numThreads - 1) ? WIDTH : (i + 1) * chunkSize;

                // Create a new thread for each column
                threads[i] = new Thread(new ImageProcessingTask(start, end, WIDTH, HEIGHT, order, mult_factor, kernel, bias, input, output));

                // Start the thread
                threads[i].start();
            }

            // Wait for all threads to finish
            try {
                for (Thread thread : threads) {
                    if (thread != null) {
                        thread.join();
                    }
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }



class ImageProcessingTask implements Runnable {
    public int start;
    public int end;
    public int HEIGHT;
    public int order;
    public float mult_factor;
    public float[][] kernel;
    public float bias;
    public BufferedImage input;
    public BufferedImage output;
    public int WIDTH;

    public ImageProcessingTask(int start, int end,int WIDTH, int HEIGHT, int order, float mult_factor, float[][] kernel, float bias, BufferedImage input, BufferedImage output) {
        this.start = start;
        this.end = end;
        this.HEIGHT = HEIGHT;
        this.order = order;
        this.mult_factor = mult_factor;
        this.kernel = kernel;
        this.bias = bias;
        this.input = input;
        this.output = output;
        this.WIDTH=WIDTH;
    }

    @Override
    public void run() {
        for (int x = start; x < end; x++) {
            for (int y = 0; y < HEIGHT; y++) {
                float red = 0f, green = 0f, blue = 0f;
                for (int i = 0; i < order; i++) {
                    for (int j = 0; j < order; j++) {
                        int imageX = (x - order / 2 + i + WIDTH) % WIDTH;
                        int imageY = (y - order / 2 + j + HEIGHT) % HEIGHT;

                        int RGB = input.getRGB(imageX, imageY);
                        int R = (RGB >> 16) & 0xff;
                        int G = (RGB >> 8) & 0xff;
                        int B = (RGB) & 0xff;

                        red += (R * kernel[i][j]);
                        green += (G * kernel[i][j]);
                        blue += (B * kernel[i][j]);
                    }
                }
                int outR, outG, outB;
                outR = Math.min(Math.max((int) (red * mult_factor + bias), 0), 255);
                outG = Math.min(Math.max((int) (green * mult_factor + bias), 0), 255);
                outB = Math.min(Math.max((int) (blue * mult_factor + bias), 0), 255);
                output.setRGB(x, y, new Color(outR, outG, outB).getRGB());
            }
        }
    }
}


