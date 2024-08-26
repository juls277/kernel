import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.border.Border;
import mpi.*;

public class Visual implements ActionListener {

    private static final int LABEL_WIDTH = 200;
    private static final int LABEL_HEIGHT = 200;

    private BufferedImage blankIm;
    private JComboBox<String> menuList;
    private JComboBox<String> processingList;
    private JButton runButton;
    private JLabel label;
    private JLabel label1;
    private JLabel timeImage;
    private JLabel processingImg;
    private JLabel preprocessingTimeLabel;
    private JFrame frame;
    private ImageIcon outputIcon;
    private String imgPath;
    private String[] _args;
    private int order;
    private float bias;
    private float factor;
    private float[][] kernel;
    private long processStart;
    private long processEnd;
    private long processOv;
    private long preprocessingTime;

    public Visual(String[] args) {
        frame = new JFrame();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);
        frame.setTitle("Kernel Image Processing");
        _args = args;
        // Use null layout manager to manually set component bounds
        frame.setLayout(null);

        preprocessingTimeLabel = new JLabel("Preprocessing time: ");
        preprocessingTimeLabel.setBounds(20, 300, 200, 30);

        processingImg = new JLabel("Convolution time: ");
        processingImg.setBounds(20, 400, 150, 30);

        label = new JLabel();
        label.setBounds(200, 250, LABEL_WIDTH, LABEL_HEIGHT);
        Border border = BorderFactory.createLineBorder(Color.BLACK, 2);
        label.setBorder(border);

        label1 = new JLabel();
        label1.setBounds(450, 250, LABEL_WIDTH, LABEL_HEIGHT);
        label1.setBorder(border);

        // Menu with filters
        String[] filterOptions = {"Sharpened Image", "Blurred Image", "Edge Detect", "Motion Blur", "Emboss", "Box blur", "Identity"};
        menuList = new JComboBox<>(filterOptions);
        menuList.setBounds(20, 50, 200, 30);

        // Processing types
        String[] processingOptions = {"Sequentional", "Parallel", "Distributive"};
        processingList = new JComboBox<>(processingOptions);
        processingList.setBounds(20, 150, 200, 30);

        // Image options
        String[] imageOptions = {"Choose Image...", "Maksim Kac", "Puppy", "500x375", "640x443", "740x416", "800x450"};
        JComboBox<String> imageList = new JComboBox<>(imageOptions);
        imageList.setBounds(20, 100, 200, 30);

        // Add listeners
        imageList.addActionListener(e -> {
            String selectedImage = (String) imageList.getSelectedItem();
            if ("Choose Image...".equals(selectedImage)) {
                chooseImageFile();
            } else {
                imgPath = getImagePath(selectedImage);
                measurePreprocessingTime();
            }
        });

        menuList.addActionListener(e -> {
            updateKernelParameters((String) menuList.getSelectedItem());
        });

        runButton = new JButton("Run");
        runButton.setBounds(50, 200, 100, 50);
        runButton.addActionListener(this);

        // Adding components
        frame.add(runButton);
        frame.add(menuList);
        frame.add(imageList);
        frame.add(processingList);
        frame.add(label);
        frame.add(label1);
        frame.add(processingImg);
        frame.add(preprocessingTimeLabel);
        frame.setVisible(true);
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == runButton) {
            if (imgPath == null || imgPath.isEmpty()) {
                JOptionPane.showMessageDialog(frame, "Please select an image.");
                return;
            }

            // Disable the run button to prevent multiple clicks
            runButton.setEnabled(false);

            // Run the task in a background thread
            new Thread(() -> {
                try {
                    int rank = MPI.COMM_WORLD.Rank();
                    int[] signal = new int[1];

                    if (rank == 0) {
                        processStart = System.currentTimeMillis();
                        signal[0] = 1; // Signal to start convolutionMPI
                    }

                    // Broadcast the signal to all processes
                    MPI.COMM_WORLD.Bcast(signal, 0, 1, MPI.INT, 0);

                    if (signal[0] == 1) {
                        Distributive.convolutionMPI(imgPath, order, factor, bias, kernel, _args);
                    }

                    if (rank == 0) {
                        processEnd = System.currentTimeMillis();
                        processOv = processEnd - processStart;
                        SwingUtilities.invokeLater(() -> {
                            processingImg.setText("Processed in " + processOv + " ms");
                            runButton.setEnabled(true);  // Re-enable the run button after processing
                        });
                        showOutputImage(Distributive.output);
                    }
                } catch (MPIException | IOException exception) {
                    exception.printStackTrace();
                }
            }).start();
        }
    }

    private void measurePreprocessingTime() {
        long startTime = System.currentTimeMillis();
        showInputImage(imgPath);
        long endTime = System.currentTimeMillis();
        preprocessingTime = endTime - startTime;
        preprocessingTimeLabel.setText("Preprocessing time: " + preprocessingTime + " ms");
    }

    private void showInputImage(String path) {
        try {
            BufferedImage inputImage = ImageIO.read(new File(path));
            Image scaledInputImage = inputImage.getScaledInstance(LABEL_WIDTH, LABEL_HEIGHT, Image.SCALE_DEFAULT);
            ImageIcon inputIcon = new ImageIcon(scaledInputImage);
            label.setIcon(inputIcon);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void showOutputImage(BufferedImage outputImage) {
        Image scaledOutputImage = outputImage.getScaledInstance(LABEL_WIDTH, LABEL_HEIGHT, Image.SCALE_DEFAULT);
        outputIcon = new ImageIcon(scaledOutputImage);
        label1.setIcon(outputIcon);
    }

    private String getImagePath(String option) {
        switch (option) {
            case "Maksim Kac":
                return "C:/Users/jusju/Desktop/kernel-image-processing-main/kernel-image-processing-master/pictures/kac.jpeg";
            case "Puppy":
                return "C:/Users/jusju/Desktop/kernel-image-processing-main/kernel-image-processing-master/pictures/puppy.jpg";
            case "500x375":
                return "C:/Users/jusju/Desktop/kernel-image-processing-main/kernel-image-processing-master/pictures/500x375.jpg";
            case "640x443":
                return "C:/Users/jusju/Desktop/kernel-image-processing-main/kernel-image-processing-master/pictures/640x443.jpg";
            case "740x416":
                return "C:/Users/jusju/Desktop/kernel-image-processing-main/kernel-image-processing-master/pictures/740x416.jpg";
            case "800x450":
                return "C:/Users/jusju/Desktop/kernel-image-processing-main/kernel-image-processing-master/pictures/800x450.jpg";
            default:
                return null;
        }
    }

    private void updateKernelParameters(String filter) {
        switch (filter) {
            case "Sharpened Image":
                order = 3;
                factor = 1f;
                bias = 0;
                kernel = new float[][]{{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};
                break;
            case "Blurred Image":
                order = 3;
                factor = 0.0625f;
                bias = 0;
                kernel = new float[][]{{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
                break;
            case "Edge Detect":
                order = 3;
                factor = 1f;
                bias = 0;
                kernel = new float[][]{{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}};
                break;
            case "Emboss":
                order = 3;
                factor = 1f;
                bias = 128;
                kernel = new float[][]{{-1, -1, 0}, {-1, 0, 1}, {0, 1, 1}};
                break;
            case "Box blur":
                order = 3;
                factor = 0.1f;
                bias = 0;
                kernel = new float[][]{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
                break;
            case "Motion Blur":
                order = 9;
                factor = 0.1111f;
                bias = 0;
                kernel = new float[][]{
                        {1, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 1, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 1, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 1, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 1, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 1, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 1, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 1, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 1}
                };
                break;
            case "Identity":
                order = 3;
                factor = 1f;
                bias = 0;
                kernel = new float[][]{{0, 0, 0}, {0, 1, 0}, {0, 0, 0}};
                break;
        }
    }

    private void chooseImageFile() {
        JFileChooser fileChooser = new JFileChooser();
        int result = fileChooser.showOpenDialog(frame);
        if (result == JFileChooser.APPROVE_OPTION) {
            File selectedFile = fileChooser.getSelectedFile();
            imgPath = selectedFile.getAbsolutePath();
            measurePreprocessingTime();
        }
    }

}
