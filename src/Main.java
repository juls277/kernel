import javax.swing.SwingUtilities;
import mpi.*;

import java.io.IOException;

public class Main {
    public static void main(String[] args) {
        MPI.Init(args);

        int rank = MPI.COMM_WORLD.Rank();

        if (rank == 0) {
            SwingUtilities.invokeLater(() -> new Visual(args));
        } else {
            listenForSignalAndProcess(args);
        }

        MPI.Finalize();
    }

    public static void listenForSignalAndProcess(String[] args) {
        try {
            int[] signal = new int[1];
            while (true) {
                MPI.COMM_WORLD.Bcast(signal, 0, 1, MPI.INT, 0);
                if (signal[0] == 1) { // 1 indicates that convolutionMPI should be called
                    System.out.println("Process " + MPI.COMM_WORLD.Rank() + ": Entering convolutionMPI.");
                    Distributive.convolutionMPI("image_path_placeholder", 3, 1.0f, 0.0f, new float[3][3], args);
                }
                // Sleep or wait until the next signal
            }
        } catch (MPIException | IOException e) {
            e.printStackTrace();
        }
    }
}
