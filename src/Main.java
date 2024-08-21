import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import javax.swing.SwingUtilities;

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
        new Visual();
    }
}

