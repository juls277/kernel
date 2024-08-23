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



    }

}
