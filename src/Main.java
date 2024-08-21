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

        // Get the total number of processes
        int size = MPI.COMM_WORLD.Size();

        // Print a message from each process
        System.out.println("Hello from process " + rank + " out of " + size);

        // Finalize the MPI environment
        MPI.Finalize();
        new Visual();
    }
}

