import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.awt.image.BufferedImage;
import mpi.*;

public class Distributive {
    public static int HEIGHT;
    public static int WIDTH;
    public static BufferedImage input, output;

    public static void convolutionMPI(String filepath, int order, float factor, float bias, float[][] kernel, String[] args) throws MPIException, IOException {
        int rank = MPI.COMM_WORLD.Rank();
        int size = MPI.COMM_WORLD.Size();

        // Define a small chunk size to optimize memory usage
        int chunkHeight = 50; // Process 50 rows at a time

        if (rank == 0) {
            // Root process reads the image and populates inputRGB
            System.out.println("Process " + rank + ": Reading image from file: " + filepath);
            input = ImageIO.read(new File(filepath));
            WIDTH = input.getWidth();
            HEIGHT = input.getHeight();
            System.out.println("Process " + rank + ": Image dimensions: " + WIDTH + "x" + HEIGHT);

            // Initialize the output image
            output = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_RGB);
        }

        // Broadcast image dimensions to all processes
        int[] dimensions = new int[2];
        if (rank == 0) {
            dimensions[0] = WIDTH;
            dimensions[1] = HEIGHT;
        }
        MPI.COMM_WORLD.Bcast(dimensions, 0, 2, MPI.INT, 0);
        WIDTH = dimensions[0];
        HEIGHT = dimensions[1];

        // Scatter and process the image in chunks
        int[] ord = new int[1];
        MPI.COMM_WORLD.Bcast(ord = new int[]{order}, 0, 1, MPI.INT, 0);
        float[] flattenedKernel = new float[ord[0] * ord[0]];
        float[] fact = new float[1];
        float[] bi = new float[1];

        // Flatten the kernel into a 1D array
        if (rank == 0) {
            int index = 0;
            for (int i = 0; i < order; i++) {
                for (int j = 0; j < order; j++) {
                    flattenedKernel[index++] = kernel[i][j];
                }
            }
        }

        MPI.COMM_WORLD.Bcast(flattenedKernel, 0, flattenedKernel.length, MPI.FLOAT, 0);
        MPI.COMM_WORLD.Bcast(fact = new float[]{factor}, 0, 1, MPI.FLOAT, 0);
        MPI.COMM_WORLD.Bcast(bi = new float[]{bias}, 0, 1, MPI.FLOAT, 0);

        for (int startRow = 0; startRow < HEIGHT; startRow += chunkHeight) {
            int currentChunkHeight = Math.min(chunkHeight, HEIGHT - startRow);

            // Allocate memory only for the current chunk
            byte[] inputChunk = new byte[WIDTH * currentChunkHeight * 3];
            byte[] outputChunk = new byte[WIDTH * currentChunkHeight * 3];
            byte[] gatheredChunk = null;

            // Root process reads the chunk and sends it to other processes
            if (rank == 0) {
                for (int y = 0; y < currentChunkHeight; y++) {
                    for (int x = 0; x < WIDTH; x++) {
                        int rgb = input.getRGB(x, startRow + y);
                        int index = (y * WIDTH + x) * 3;
                        inputChunk[index] = (byte) ((rgb >> 16) & 0xFF); // Red
                        inputChunk[index + 1] = (byte) ((rgb >> 8) & 0xFF); // Green
                        inputChunk[index + 2] = (byte) (rgb & 0xFF); // Blue
                    }
                }
                gatheredChunk = new byte[WIDTH * currentChunkHeight * 3 * size];
            }

            // Broadcast the chunk to all processes
            MPI.COMM_WORLD.Bcast(inputChunk, 0, inputChunk.length, MPI.BYTE, 0);

            // Perform computation on the current chunk
            for (int y = 0; y < currentChunkHeight; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    float red = 0f, green = 0f, blue = 0f;
                    for (int i = 0; i < ord[0]; i++) {
                        for (int j = 0; j < ord[0]; j++) {
                            int imageX = (x - ord[0] / 2 + j + WIDTH) % WIDTH;
                            int imageY = (y - ord[0] / 2 + i + currentChunkHeight) % currentChunkHeight;

                            int pixelIndex = (imageY * WIDTH + imageX) * 3;
                            int R = inputChunk[pixelIndex] & 0xFF;
                            int G = inputChunk[pixelIndex + 1] & 0xFF;
                            int B = inputChunk[pixelIndex + 2] & 0xFF;

                            red += (R * flattenedKernel[i * ord[0] + j]);
                            green += (G * flattenedKernel[i * ord[0] + j]);
                            blue += (B * flattenedKernel[i * ord[0] + j]);
                        }
                    }
                    int outR = Math.min(Math.max((int) (red * fact[0] + bi[0]), 0), 255);
                    int outG = Math.min(Math.max((int) (green * fact[0] + bi[0]), 0), 255);
                    int outB = Math.min(Math.max((int) (blue * fact[0] + bi[0]), 0), 255);
                    int outputIndex = (y * WIDTH + x) * 3;
                    outputChunk[outputIndex] = (byte) outR;
                    outputChunk[outputIndex + 1] = (byte) outG;
                    outputChunk[outputIndex + 2] = (byte) outB;
                }
            }

            // Gather the results back to the root process for the current chunk
            MPI.COMM_WORLD.Gather(outputChunk, 0, outputChunk.length, MPI.BYTE, gatheredChunk, 0, outputChunk.length, MPI.BYTE, 0);

            // Assemble the output image on the root process
            if (rank == 0) {
                for (int y = 0; y < currentChunkHeight; y++) {
                    for (int x = 0; x < WIDTH; x++) {
                        int outputIndex = (y * WIDTH + x) * 3;
                        int rgb = ((gatheredChunk[outputIndex] & 0xFF) << 16) |
                                ((gatheredChunk[outputIndex + 1] & 0xFF) << 8) |
                                (gatheredChunk[outputIndex + 2] & 0xFF);
                        output.setRGB(x, startRow + y, rgb);
                    }
                }
            }
        }

        // Save the output image on the root process
        if (rank == 0) {
            String outputFilepath = "output.png";
            ImageIO.write(output, "png", new File(outputFilepath));
            System.out.println("Process " + rank + ": Output image saved as " + outputFilepath);
        }

        System.out.println("Process " + rank + ": Finalizing MPI.");
    }
}
