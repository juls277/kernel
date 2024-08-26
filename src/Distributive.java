/*import javax.imageio.ImageIO;
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

        int[] inputRGB = null;
        int[] recvBuffer = null;
        int[] outputRGB = null;

        if (rank == 0) {
            // Root process reads the image and populates inputRGB
            System.out.println("Process " + rank + ": Reading image from file: " + filepath);
            input = ImageIO.read(new File(filepath));
            WIDTH = input.getWidth();
            HEIGHT = input.getHeight();
            System.out.println("Process " + rank + ": Image dimensions: " + WIDTH + "x" + HEIGHT);

            inputRGB = new int[WIDTH * HEIGHT];
            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    inputRGB[y * WIDTH + x] = input.getRGB(x, y);
                }
            }
            System.out.println("Process " + rank + ": Image flattened into 1D array.");
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
            inputRGB = new int[WIDTH * HEIGHT];
        }

        // Broadcast the populated inputRGB array from root to all processes
        MPI.COMM_WORLD.Bcast(inputRGB, 0, WIDTH * HEIGHT, MPI.INT, 0);
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
            sendCounts[i] = WIDTH * (endRow - startRow);
            displs[i] = WIDTH * startRow;
        }

        recvBuffer = new int[sendCounts[rank]];
        outputRGB = new int[recvBuffer.length];

        // Scatter the image data to each process
        MPI.COMM_WORLD.Scatterv(inputRGB, 0, sendCounts, displs, MPI.INT, recvBuffer, 0, recvBuffer.length, MPI.INT, 0);
        System.out.println("Process " + rank + ": Received image data chunk with inputRGB [0]> " + inputRGB[0]);

        // Flatten the kernel into a 1D array
        float[] flattenedKernel = new float[order * order];
        int index = 0;
        for (int i = 0; i < order; i++) {
            for (int j = 0; j < order; j++) {
                flattenedKernel[index++] = kernel[i][j];
            }
        }
        System.out.println("Process " + rank + ": Kernel flattened.");

        // Perform computation on chunk
        System.out.println("Process " + rank + ": Starting convolution."+ "starting row, ending row" + startingRow + " " + endingRow);
        for (int y = 0; y < (endingRow - startingRow); y++) {
            for (int x = 0; x < WIDTH; x++) {

                float red = 0f, green = 0f, blue = 0f;
                for (int i = 0; i < order; i++) {
                    for (int j = 0; j < order; j++) {
                        int imageX = (x - order / 2 + j + WIDTH) % WIDTH;
                        int imageY = (y - order / 2 + i + (endingRow - startingRow)) % (endingRow - startingRow);

                        int RGB = recvBuffer[imageY * WIDTH + imageX];
                        int R = (RGB >> 16) & 0xff;
                        int G = (RGB >> 8) & 0xff;
                        int B = (RGB) & 0xff;

                        red += (R * flattenedKernel[i * order + j]);
                        green += (G * flattenedKernel[i * order + j]);
                        blue += (B * flattenedKernel[i * order + j]);
                    }
                }
                int outR = Math.min(Math.max((int) (red * factor + bias), 0), 255);
                int outG = Math.min(Math.max((int) (green * factor + bias), 0), 255);
                int outB = Math.min(Math.max((int) (blue * factor + bias), 0), 255);
                outputRGB[y * WIDTH + x] = (outR << 16) | (outG << 8) | outB;
            }
        }
        System.out.println("Process " + rank + " outputRGB Length>  " + outputRGB.length);

        System.out.println("Process " + rank + ": Finished convolution.");
        /*int[] outputCombinedRGB = new int[inputRGB.length];
        // Gather chunks back to the root process (rank 0) using Gatherv
        MPI.COMM_WORLD.Gatherv(outputRGB, 0, outputRGB.length, MPI.INT, outputRGB, 0, sendCounts, displs, MPI.INT, 0);
        System.out.println("Process " + rank + ": gathered chunks using Gatherv.");

        // Only the root process assembles and writes the final image
        if (rank == 0) {
            // Initialize output image on the root process
            output = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_RGB);

            System.out.println("Process " + rank + ": Assembling final image.");
            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    output.setRGB(x, y, outputCombinedRGB[y * WIDTH + x]);
                }
            }
            System.out.println("Process " + rank + ": Writing output image to file.");
            ImageIO.write(output, "jpg", new File("output.jpg"));
        }
        if (rank == 0) {
            output = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_RGB);
            int[] fullOutputRGB = new int[WIDTH * HEIGHT];
            MPI.COMM_WORLD.Gatherv(outputRGB, 0, outputRGB.length, MPI.INT, fullOutputRGB, 0, sendCounts, displs, MPI.INT, 0);

            // Root process populates the output image
            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    output.setRGB(x, y, fullOutputRGB[y * WIDTH + x]);
                }
            }

            // Save the output image
            String outputFilepath = "output.png";
            ImageIO.write(output, "png", new File(outputFilepath));
            System.out.println("Process " + rank + ": Output image saved as " + outputFilepath);
        } else {
            MPI.COMM_WORLD.Gatherv(outputRGB, 0, outputRGB.length, MPI.INT, null, 0, null, null, MPI.INT, 0);
        }

        MPI.COMM_WORLD.Barrier();
        System.out.println("Process " + rank + ": Finalizing MPI.");
        MPI.Finalize();
    }
}*/
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

        int[] inputRGB = null;
        int[] recvBuffer = null;
        int[] outputRGB = null;

        if (rank == 0) {
            // Root process reads the image and populates inputRGB
            System.out.println("Process " + rank + ": Reading image from file: " + filepath);
            input = ImageIO.read(new File(filepath));
            WIDTH = input.getWidth();
            HEIGHT = input.getHeight();
            System.out.println("Process " + rank + ": Image dimensions: " + WIDTH + "x" + HEIGHT);

            inputRGB = new int[WIDTH * HEIGHT];
            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    inputRGB[y * WIDTH + x] = input.getRGB(x, y);
                }
            }
            System.out.println("Process " + rank + ": Image flattened into 1D array.");
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
            inputRGB = new int[WIDTH * HEIGHT];

        }

        // Broadcast the populated inputRGB array from root to all processes
        MPI.COMM_WORLD.Bcast(inputRGB, 0, WIDTH * HEIGHT, MPI.INT, 0);
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
            sendCounts[i] = WIDTH * (endRow - startRow);
            displs[i] = WIDTH * startRow;
        }

        recvBuffer = new int[sendCounts[rank]];
        outputRGB = new int[recvBuffer.length];

        // Scatter the image data to each process
        MPI.COMM_WORLD.Scatterv(inputRGB, 0, sendCounts, displs, MPI.INT, recvBuffer, 0, recvBuffer.length, MPI.INT, 0);
        System.out.println("Process " + rank + ": Received image data chunk with inputRGB [0]> " + inputRGB[0]);

        // Flatten the kernel into a 1D array
        float[] flattenedKernel = new float[order * order];
        int index = 0;
        for (int i = 0; i < order; i++) {
            for (int j = 0; j < order; j++) {
                flattenedKernel[index++] = kernel[i][j];
            }
        }
        System.out.println("Process " + rank + ": Kernel flattened.");

        // Perform computation on chunk
        System.out.println("Process " + rank + ": Starting convolution." + "starting row, ending row" + startingRow + " " + endingRow);
        for (int y = 0; y < (endingRow - startingRow); y++) {
            for (int x = 0; x < WIDTH; x++) {

                float red = 0f, green = 0f, blue = 0f;
                for (int i = 0; i < order; i++) {
                    for (int j = 0; j < order; j++) {
                        int imageX = (x - order / 2 + j + WIDTH) % WIDTH;
                        int imageY = (y - order / 2 + i + (endingRow - startingRow)) % (endingRow - startingRow);

                        int RGB = recvBuffer[imageY * WIDTH + imageX];
                        int R = (RGB >> 16) & 0xff;
                        int G = (RGB >> 8) & 0xff;
                        int B = (RGB) & 0xff;

                        red += (R * flattenedKernel[i * order + j]);
                        green += (G * flattenedKernel[i * order + j]);
                        blue += (B * flattenedKernel[i * order + j]);
                    }
                }
                int outR = Math.min(Math.max((int) (red * factor + bias), 0), 255);
                int outG = Math.min(Math.max((int) (green * factor + bias), 0), 255);
                int outB = Math.min(Math.max((int) (blue * factor + bias), 0), 255);
                outputRGB[y * WIDTH + x] = (outR << 16) | (outG << 8) | outB;
                System.out.println("Process " + rank + "outputRGB  " + outputRGB[y * WIDTH + x]);
            }
        }
        System.out.println("Process " + rank + " outputRGB Length>  " + outputRGB.length);

        System.out.println("Process " + rank + ": Finished convolution.");

        // Gather the results back to the root process
        int[] fullOutputRGB = null;
        fullOutputRGB = new int[WIDTH * HEIGHT];


        MPI.COMM_WORLD.Gatherv(outputRGB, 0, outputRGB.length, MPI.INT, fullOutputRGB, 0, sendCounts, displs, MPI.INT, 0);
        // Broadcast the gathered fullOutputRGB to all processes
        MPI.COMM_WORLD.Bcast(fullOutputRGB, 0, WIDTH * HEIGHT, MPI.INT, 0);
         // All processes now have the complete image data
        if (rank ==0) {
            output = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_RGB);
            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    output.setRGB(x, y, fullOutputRGB[y * WIDTH + x]);
                    //System.out.println("Process " + rank + ": Output: " + output.getRGB(x, y));
                }
            }
        }

        if (rank == 0) {
            // Save the output image
            String outputFilepath = "output.png";
            ImageIO.write(output, "png", new File(outputFilepath));
            System.out.println("Process " + rank + ": Output image saved as " + outputFilepath);

        } //MPI.Finalize();
        }
    }

