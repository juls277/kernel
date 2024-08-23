import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import javax.swing.SwingUtilities;
import mpi.*;
import mpjdev.*;
import xdev.*;

public class Main {
    public static BufferedImage input;
    public static int WIDTH;
    public static int HEIGHT;

    // Method to load an image from the given filename
    public static void loadImage(String filename) throws IOException {
        input = ImageIO.read(new File(filename));
        WIDTH = input.getWidth();
        HEIGHT = input.getHeight();
    }

    // Main method to launch the GUI
    public static void main(String[] args) {
        MPI.Init(args);

        // Get the rank of the current process
        int rank = MPI.COMM_WORLD.Rank();

        // Define the root process that will broadcast the value (usually rank 0)
        int root = 0;

        // Create a buffer to hold the value to be broadcasted
        int[] value = new int[1];

        if (rank == root) {
            // If this is the root process, initialize the value
            value[0] = 42;
            System.out.println("Root process (rank " + rank + ") broadcasting value: " + value[0]);
        }

        // Broadcast the value from the root process to all other processes
        MPI.COMM_WORLD.Bcast(value, 0, 1, MPI.INT, root);

        // Print the received value on each process
        System.out.println("Process " + rank + " received value: " + value[0]);

        // Finalize the MPI environment
        MPI.Finalize();
        new Visual();
    }
}

