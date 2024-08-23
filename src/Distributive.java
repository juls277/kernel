import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import java.awt.Color;
import java.awt.image.BufferedImage;
import mpi.*;

public class Distributive {
    public static int HEIGHT;
    public static int WIDTH;
    public static BufferedImage input, output;

    public static void convolutionMPI(String filepath, int order, float factor, float bias, float[][] kernel)throws MPIException, IOException {
        input = ImageIO.read(new File(filepath));

        WIDTH = input.getWidth();
        HEIGHT = input.getHeight();

        int rank = MPI.COMM_WORLD.Rank();
        int size = MPI.COMM_WORLD.Size();

        //assign certain chunk to each process
        int rowsPerProcess = HEIGHT / size;
        int startingRow = rank * rowsPerProcess;
        int endingRow = (rank == size - 1) ? HEIGHT : startingRow + rowsPerProcess;

        int[] inputRGB = null;
        int[] outputRGB = new int[WIDTH * (endingRow - startingRow)];

        //flatten image into 1D array to ease sending and receiving messages in mpi
        if (rank == 0) {
            // Flatten the input image's RGB data
            inputRGB = new int[WIDTH * HEIGHT];
            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    inputRGB[y * WIDTH + x] = input.getRGB(x, y);
                }
            }
        }

        int[] recvBuffer = new int[WIDTH * (endingRow - startingRow)];
        //scatter parts of the image to processes
        MPI.COMM_WORLD.Scatter(inputRGB, startingRow * WIDTH, recvBuffer.length, MPI.INT, recvBuffer, 0, recvBuffer.length, MPI.INT, 0);
        //perform computation on chunck
        for (int y = 0; y < endingRow - startingRow; y++) {
            for (int x = 0; x < WIDTH; x++) {
                float red = 0f, green = 0f, blue = 0f;
                for (int i = 0; i < order; i++) {
                    for (int j = 0; j < order; j++) {
                        int imageX = (x - order / 2 + i + WIDTH) % WIDTH;
                        int imageY = (y - order / 2 + j + HEIGHT) % HEIGHT;

                        int RGB = recvBuffer[imageY * WIDTH + imageX];
                        int R = (RGB >> 16) & 0xff;
                        int G = (RGB >> 8) & 0xff;
                        int B = (RGB) & 0xff;

                        red += (R * kernel[i][j]);
                        green += (G * kernel[i][j]);
                        blue += (B * kernel[i][j]);
                    }
                }
                int outR = Math.min(Math.max((int)(red * factor + bias), 0), 255);
                int outG = Math.min(Math.max((int)(green * factor + bias), 0), 255);
                int outB = Math.min(Math.max((int)(blue * factor + bias), 0), 255);
                outputRGB[y * WIDTH + x] = (outR << 16) | (outG << 8) | outB;
            }
        }

        //gather chunks back to a single image
        MPI.COMM_WORLD.Gather(outputRGB, 0, outputRGB.length, MPI.INT, inputRGB, startingRow * WIDTH, outputRGB.length, MPI.INT, 0);

        //master process transforms 1D arr back to 2D and writes output
        if (rank == 0) {
            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    output.setRGB(x, y, inputRGB[y * WIDTH + x]);
                }
            }
        }

    }

}
