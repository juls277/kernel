import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import static java.lang.System.out;

public class Sequentional {
    public static int HEIGHT;
    public static int WIDTH;
    public static long processStart;
    public static long processEnd;
    public static long processOv;
    public static BufferedImage input,output;

    public static void convolutionImage(String filepath, int order, float factor, float bias, float[][] kernel)throws IOException {
        input = ImageIO.read(new File(filepath));

        WIDTH = input.getWidth();
        HEIGHT = input.getHeight();




        output = new BufferedImage(WIDTH, HEIGHT, input.getType());
        // float[][] kernel; = new float[order][order];
        float sum_of_elements = 0.0f;
        float mult_factor = factor;

        for(int i=0; i < order; i++)
        {
            for(int j=0; j < order; j++)
            {
                out.print("\t"+kernel[i][j]);
                sum_of_elements += kernel[i][j];
            }
            //out.println();
        }


        for(int x=0;x<WIDTH;x++)
        {
            for(int y=0;y<HEIGHT;y++)
            {
                float red=0f,green=0f,blue=0f;
                for(int i=0;i<order;i++)
                {
                    for(int j=0;j<order;j++)
                    {

                        int imageX = (x - order / 2 + i + WIDTH) % WIDTH;
                        int imageY = (y - order / 2 + j + HEIGHT) % HEIGHT;

                        int RGB = input.getRGB(imageX,imageY);
                        int R = (RGB >> 16) & 0xff; // Red Value
                        int G = (RGB >> 8) & 0xff;	// Green Value
                        int B = (RGB) & 0xff;		// Blue Value


                        red += (R*kernel[i][j]);
                        green += (G*kernel[i][j]);
                        blue += (B*kernel[i][j]);
                    }
                }
                int outR, outG, outB;

                outR = Math.min(Math.max((int)(red*mult_factor),0),255);
                outG = Math.min(Math.max((int)(green*mult_factor),0),255);
                outB = Math.min(Math.max((int)(blue*mult_factor),0),255);
                // Pixel is written to output image
                output.setRGB(x,y,new Color(outR,outG,outB).getRGB());
            }
        }
      //  processEnd=System.currentTimeMillis();
      //  processOv=processEnd-processStart;
    }
}
