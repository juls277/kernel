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

        String[] filterOptions = {"Sharpened Image", "Blurred Image", "Edge Detect", "Emboss", "Gaussian Blur 5x5", "Unsharp Masking 5x5", "Motion Blur", "Identity"};
        menuList = new JComboBox<>(filterOptions);
        menuList.setBounds(20, 50, 200, 30);

        String[] processingOptions = {"Sequentional", "Parallel", "Distributive"};
        processingList = new JComboBox<>(processingOptions);
        processingList.setBounds(20, 150, 200, 30);

        String[] imageOptions = {"Choose Image...", "450x300", "600x400", "1350x900", "1500x1000", "1920x1280", "2250x1500", "3000x2000", "4500x3000", "6000x4000", "7500x5000"};
        JComboBox<String> imageList = new JComboBox<>(imageOptions);
        imageList.setBounds(20, 100, 200, 30);

        imageList.addActionListener(e -> {
            String selectedImage = (String) imageList.getSelectedItem();
            if ("Choose Image...".equals(selectedImage)) {
                chooseImageFile();
            } else {
                imgPath = getImagePath(selectedImage);
                clearPreviousOutputs(); // Clear previous outputs and processing times
                measurePreprocessingTime();
            }
        });

        menuList.addActionListener(e -> {
            updateKernelParameters((String) menuList.getSelectedItem());
            clearOutputAndProcessingTime(); // Clear previous output and processing time when a new filter is selected
        });

        processingList.addActionListener(e -> {
            clearOutputAndProcessingTime(); // Clear previous output and processing time when a new processing mode is selected
        });

        runButton = new JButton("Run");
        runButton.setBounds(50, 200, 100, 50);
        runButton.addActionListener(this);

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

            runButton.setEnabled(false);

            String processingType = (String) processingList.getSelectedItem();

            new Thread(() -> {
                try {
                    switch (processingType) {
                        case "Sequentional":
                            processStart = System.currentTimeMillis();
                            runSequential();
                            processEnd = System.currentTimeMillis();
                            break;

                        case "Parallel":
                            processStart = System.currentTimeMillis();
                            runParallel();
                            processEnd = System.currentTimeMillis();
                            break;

                        case "Distributive":
                            int rank = MPI.COMM_WORLD.Rank();
                            int[] signal = new int[1];

                            if (rank == 0) {
                                processStart = System.currentTimeMillis();
                                signal[0] = 1;
                            }

                            MPI.COMM_WORLD.Bcast(signal, 0, 1, MPI.INT, 0);

                            if (signal[0] == 1) {
                                Distributive.convolutionMPI(imgPath, order, factor, bias, kernel, _args);
                            }

                            if (rank == 0) {
                                processEnd = System.currentTimeMillis();
                                SwingUtilities.invokeLater(() -> {
                                    processingImg.setText("Processed in " + processOv + " ms");
                                    runButton.setEnabled(true);
                                });
                                showOutputImage(Distributive.output);
                            }
                            break;
                    }

                    processOv = processEnd - processStart;
                    SwingUtilities.invokeLater(() -> {
                        processingImg.setText("Processed in " + processOv + " ms");
                        runButton.setEnabled(true);
                    });

                } catch (MPIException | IOException exception) {
                    exception.printStackTrace();
                }
            }).start();
        }
    }

    private void runSequential() throws IOException {
        Sequentional.convolutionImage(imgPath, order, factor, bias, kernel);
        BufferedImage outputImage = Sequentional.output;
        showOutputImage(outputImage);
    }

    private void runParallel() throws IOException {
        Parallel.convolutionImage(imgPath, order, factor, bias, kernel);
        BufferedImage outputImage = Parallel.output;
        showOutputImage(outputImage);
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
        if (outputImage == null) {
            label1.setIcon(null); // Clear the output label
            return;
        }
        Image scaledOutputImage = outputImage.getScaledInstance(LABEL_WIDTH, LABEL_HEIGHT, Image.SCALE_DEFAULT);
        outputIcon = new ImageIcon(scaledOutputImage);
        label1.setIcon(outputIcon);
    }

    private void clearPreviousOutputs() {
        clearOutputAndProcessingTime(); // Clear the output label and processing time
        preprocessingTimeLabel.setText("Preprocessing time: "); // Reset preprocessing time label only when a new image is selected
    }

    private void clearOutputAndProcessingTime() {
        clearOutputImage(); // Clear the output label
        processingImg.setText("Convolution time: "); // Reset processing time label
    }

    private void clearOutputImage() {
        label1.setIcon(null); // Clear the output label
    }

    private String getImagePath(String option) {
        switch (option) {
            case "450x300":
                return "C:/Users/jusju/Desktop/kernel-image-processing-main/kernel-image-processing-master/pictures/450x300.jpg";
            case "600x400":
                return "C:/Users/jusju/Desktop/kernel-image-processing-main/kernel-image-processing-master/pictures/600x400.jpg";
            case "1350x900":
                return "C:/Users/jusju/Desktop/kernel-image-processing-main/kernel-image-processing-master/pictures/1350x900.jpg";
            case "1500x1000":
                return "C:/Users/jusju/Desktop/kernel-image-processing-main/kernel-image-processing-master/pictures/1500x1000.jpg";
            case "1920x1280":
                return "C:/Users/jusju/Desktop/kernel-image-processing-main/kernel-image-processing-master/pictures/1920x1280.jpg";
            case "2250x1500":
                return "C:/Users/jusju/Desktop/kernel-image-processing-main/kernel-image-processing-master/pictures/2250x1500.jpg";
            case "3000x2000":
                return "C:/Users/jusju/Desktop/kernel-image-processing-main/kernel-image-processing-master/pictures/3000x2000.jpg";
            case "4500x3000":
                return "C:/Users/jusju/Desktop/kernel-image-processing-main/kernel-image-processing-master/pictures/4500x3000.jpg";
            case "6000x4000":
                return "C:/Users/jusju/Desktop/kernel-image-processing-main/kernel-image-processing-master/pictures/6000x4000.jpg";
            case "7500x5000":
                return "C:/Users/jusju/Desktop/kernel-image-processing-main/kernel-image-processing-master/pictures/7500x5000.jpg";
            default:
                return null;
        }
    }

    private void updateKernelParameters(String filter) {
        switch (filter) {
            case "Sharpened Image":
                order = 3;
                factor = 1.9f;
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
                factor = 0.4f;
                bias = 0;
                kernel = new float[][]{{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}};
                break;

            case "Emboss":
                order = 3;
                factor = 1f;
                bias = 0;
                kernel = new float[][]{{-1, -1, 0}, {-1, 0, 1}, {0, 1, 1}};
                break;

            case "Gaussian Blur 5x5":
                order = 5;
                factor = 1f / 256f;
                bias = 0;
                kernel = new float[][]{
                        {1, 4, 6, 4, 1},
                        {4, 16, 24, 16, 4},
                        {6, 24, 36, 24, 6},
                        {4, 16, 24, 16, 4},
                        {1, 4, 6, 4, 1}
                };
                break;

            case "Motion Blur":
                order = 5;
                factor = 0.2f;
                bias = 0;
                kernel = new float[][]{
                        {1, 0, 0, 0, 0},
                        {0, 1, 0, 0, 0},
                        {0, 0, 1, 0, 0},
                        {0, 0, 0, 1, 0},
                        {0, 0, 0, 0, 1}
                };
                break;

            case "Unsharp Masking 5x5":
                order = 5;
                factor = -(1f / 512f);
                bias = 0;
                kernel = new float[][]{
                        {1, 4, 6, 4, 1},
                        {4, 16, 24, 16, 4},
                        {6, 24, -476, 24, 6},
                        {4, 16, 24, 16, 4},
                        {1, 4, 6, 4, 1}
                };
                break;

            case "Identity":
                order = 3;
                factor = 1.0f;
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
            clearPreviousOutputs(); // Clear previous outputs and processing times
            measurePreprocessingTime();
        }
    }
}
