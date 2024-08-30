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

        byte[] inputRGB = null;
        byte[] recvBuffer = null;
        byte[] outputRGB = null;

        if (rank == 0) {
            // Root process reads the image and populates inputRGB
            System.out.println("Process " + rank + ": Reading image from file: " + filepath);
            input = ImageIO.read(new File(filepath));
            WIDTH = input.getWidth();
            HEIGHT = input.getHeight();
            System.out.println("Process " + rank + ": Image dimensions: " + WIDTH + "x" + HEIGHT);

            inputRGB = new byte[WIDTH * HEIGHT * 3]; // 3 bytes per pixel (R, G, B)
            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    int rgb = input.getRGB(x, y);
                    int index = (y * WIDTH + x) * 3;
                    inputRGB[index] = (byte) ((rgb >> 16) & 0xFF); // Red
                    inputRGB[index + 1] = (byte) ((rgb >> 8) & 0xFF); // Green
                    inputRGB[index + 2] = (byte) (rgb & 0xFF); // Blue
                }
            }
            System.out.println("Process " + rank + ": Image flattened into 1D byte array.");
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

        // Allocate memory for inputRGB on all processes
        if (rank != 0) {
            inputRGB = new byte[WIDTH * HEIGHT * 3];
        }

        // Broadcast the populated inputRGB array from root to all processes
        MPI.COMM_WORLD.Bcast(inputRGB, 0, WIDTH * HEIGHT * 3, MPI.BYTE, 0);
        System.out.println("Process " + rank + ": Received broadcasted inputRGB array.");

        // Define the send counts and displacements for Scatterv
        int[] sendCounts = new int[size];
        int[] displs = new int[size];

        int rowsPerProcess = HEIGHT / size;
        int extraRows = HEIGHT % size;
        int startingRow = rank * rowsPerProcess + Math.min(rank, extraRows);
        int endingRow = startingRow + rowsPerProcess + (rank < extraRows ? 1 : 0);

        for (int i = 0; i < size; i++) {
            int startRow = i * rowsPerProcess + Math.min(i, extraRows);
            int endRow = startRow + rowsPerProcess + (i < extraRows ? 1 : 0);
            sendCounts[i] = WIDTH * (endRow - startRow) * 3;
            displs[i] = WIDTH * startRow * 3;
        }

        recvBuffer = new byte[sendCounts[rank]];
        outputRGB = new byte[recvBuffer.length];
        int[] ord = new int[1];
        MPI.COMM_WORLD.Bcast(ord = new int[]{order}, 0, 1, MPI.INT, 0);
        float[] flattenedKernel = new float[ord[0] * ord[0]];
        float[] fact = new float[1];
        float[] bi = new float[1];

        // Flatten the kernel into a 1D array
        if (rank == 0) {
            flattenedKernel = new float[order * order];
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

        // Scatter the image data to each process
        MPI.COMM_WORLD.Scatterv(inputRGB, 0, sendCounts, displs, MPI.BYTE, recvBuffer, 0, recvBuffer.length, MPI.BYTE, 0);
        System.out.println("Process " + rank + ": Received image data chunk.");

        // Perform computation on chunk
        System.out.println("Process " + rank + ": Starting convolution." + " starting row, ending row " + startingRow + " " + endingRow);
        for (int y = 0; y < (endingRow - startingRow); y++) {
            for (int x = 0; x < WIDTH; x++) {

                float red = 0f, green = 0f, blue = 0f;
                for (int i = 0; i < ord[0]; i++) {
                    for (int j = 0; j < ord[0]; j++) {
                        int imageX = (x - ord[0] / 2 + j + WIDTH) % WIDTH;
                        int imageY = (y - ord[0] / 2 + i + (endingRow - startingRow)) % (endingRow - startingRow);

                        int pixelIndex = (imageY * WIDTH + imageX) * 3;
                        int R = recvBuffer[pixelIndex] & 0xFF;
                        int G = recvBuffer[pixelIndex + 1] & 0xFF;
                        int B = recvBuffer[pixelIndex + 2] & 0xFF;

                        red += (R * flattenedKernel[i * ord[0] + j]);
                        green += (G * flattenedKernel[i * ord[0] + j]);
                        blue += (B * flattenedKernel[i * ord[0] + j]);
                    }
                }
                int outR = Math.min(Math.max((int) (red * fact[0] + bi[0]), 0), 255);
                int outG = Math.min(Math.max((int) (green * fact[0] + bi[0]), 0), 255);
                int outB = Math.min(Math.max((int) (blue * fact[0] + bi[0]), 0), 255);
                int outputIndex = (y * WIDTH + x) * 3;
                outputRGB[outputIndex] = (byte) outR;
                outputRGB[outputIndex + 1] = (byte) outG;
                outputRGB[outputIndex + 2] = (byte) outB;
            }
        }
        System.out.println("Process " + rank + " outputRGB Length>  " + outputRGB.length);

        System.out.println("Process " + rank + ": Finished convolution.");

        // Gather the results back to the root process
        byte[] fullOutputRGB = new byte[WIDTH * HEIGHT * 3];

        MPI.COMM_WORLD.Gatherv(outputRGB, 0, outputRGB.length, MPI.BYTE, fullOutputRGB, 0, sendCounts, displs, MPI.BYTE, 0);

        // Save the output image on the root process
        if (rank == 0) {
            output = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_RGB);
            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    int outputIndex = (y * WIDTH + x) * 3;
                    int rgb = ((fullOutputRGB[outputIndex] & 0xFF) << 16) |
                            ((fullOutputRGB[outputIndex + 1] & 0xFF) << 8) |
                            (fullOutputRGB[outputIndex + 2] & 0xFF);
                    output.setRGB(x, y, rgb);
                }
            }

            String outputFilepath = "output.png";
            ImageIO.write(output, "png", new File(outputFilepath));
            System.out.println("Process " + rank + ": Output image saved as " + outputFilepath);
        }

        System.out.println("Process " + rank + ": Finalizing MPI.");
    }
}
