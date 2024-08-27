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
                    inputRGB[y * WIDTH + x] = input.getRGB(x,y);
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
        System.out.println("Process " + rank + ": Received broadcasted inputRGB array." + inputRGB[70000]);

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
        System.out.println("Process " + rank + ": Received image data chunk.");

        // Flatten the kernel into a 1D array
        float[] flattenedKernel = new float[order * order];
        int index = 0;
        for (int i = 0; i < order; i++) {
            for (int j = 0; j < order; j++) {
                flattenedKernel[index++] = kernel[i][j];
               // System.out.println("Process " + rank + " got kernel " + kernel[i][j] + ".");
            }
        }
        //System.out.println("Process " + rank + ": Kernel flattened.");
        /*MPI.COMM_WORLD.Bcast(flattenedKernel,0, flattenedKernel.length, MPI.FLOAT, 0);
        MPI.COMM_WORLD.Bcast(order,0,1, MPI.INT, 0);
        MPI.COMM_WORLD.Bcast(factor,0,1, MPI.FLOAT, 0);
        MPI.COMM_WORLD.Bcast(bias,0,1, MPI.FLOAT, 0);*/
        MPI.COMM_WORLD.Bcast(flattenedKernel, 0, flattenedKernel.length, MPI.FLOAT, 0);
        MPI.COMM_WORLD.Bcast(new int[]{order}, 0, 1, MPI.INT, 0);
        MPI.COMM_WORLD.Bcast(new float[]{factor}, 0, 1, MPI.FLOAT, 0);
        MPI.COMM_WORLD.Bcast(new float[]{bias}, 0, 1, MPI.FLOAT, 0);
        // Perform computation on chunk
        System.out.println("Process " + rank + ": Starting convolution." + " starting row, ending row " + startingRow + " " + endingRow);
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

        // Gather the results back to the root process
        int[] fullOutputRGB = new int[WIDTH * HEIGHT];
        /*if (rank == 0) {
             fullOutputRGB = new int[WIDTH * HEIGHT];
        } */

        MPI.COMM_WORLD.Gatherv(outputRGB, 0, outputRGB.length, MPI.INT, fullOutputRGB, 0, sendCounts, displs, MPI.INT, 0);

        // Broadcast the gathered fullOutputRGB to all processes
        MPI.COMM_WORLD.Bcast(fullOutputRGB, 0, WIDTH * HEIGHT, MPI.INT, 0);

        // Verify that all processes received the fullOutputRGB correctly
        System.out.println("Process " + rank + ": Verifying fullOutputRGB after Bcast:");
        if (rank == 0) {
            for (int i = 70000; i < Math.min(70020, fullOutputRGB.length); i++) {  // Print only first 10 elements for brevity
                System.out.println("Process " + rank + ": fullOutputRGB[" + i + "] = " + fullOutputRGB[i]);
            }
        }

        System.out.println("Width is " + WIDTH);
        System.out.println("Height is " + HEIGHT);
        MPI.COMM_WORLD.Bcast(fullOutputRGB, 0, WIDTH * HEIGHT, MPI.INT, 0);
        //output = new BufferedImage;
        if (rank == 0) {
            output = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_RGB);
            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    // output.setRGB(x, y, fullOutputRGB[y * WIDTH + x]);
                    //  System.out.println("RGB pixel at " + x + " " + y + " " + output.getRGB(x, y));
                        // int rgb = input.getRGB();
                        output.setRGB(x, y, fullOutputRGB[y * WIDTH + x]);



                }
            }

        }
        /*System.out.println("After 2D conversion:");
        for (int y = 0; y < HEIGHT; y++) {  // Check only first 2 rows
            for (int x = 0; x < WIDTH; x++) {  // Check only first 5 columns
                int rgb = output.getRGB(x, y);
                System.out.println("output.getRGB(" + x + "," + y + ") = " + rgb);
            }
        }*/
        // Save the output image
        if (rank == 0) {
            String outputFilepath = "output.png";
            ImageIO.write(output, "png", new File(outputFilepath));
            System.out.println("Process " + rank + ": Output image saved as " + outputFilepath);
        }

        // Print fullOutputRGB for verification after gathering
        //System.out.println("Process " + rank + ": Full output RGB after Gatherv:");
           /* for (int i = 0; i < fullOutputRGB.length; i++) {
                System.out.println(fullOutputRGB[i]);
            }*/
        // }

        // Finalize the MPI environment
        //MPI.COMM_WORLD.Barrier();
        System.out.println("Process " + rank + ": Finalizing MPI.");
        //MPI.Finalize();
    }
}