package AnimTests;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

public class BlurAndZoomOut {

	public static void main(String[] args) throws IOException {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		BufferedImage in = ImageIO.read(new File("raw/2/2.jpg"));
		Mat inMat = CoreSystems.bufferedImageToMat(in);
		Imgproc.cvtColor(inMat, inMat, Imgproc.COLOR_BGR2GRAY);
		int blur = 400;
		for(int i = 1; i <= 540; i ++) {
			System.out.println(i);
			Mat out = new Mat();
			if(blur - i > 0) {
				Imgproc.blur(inMat, out, new Size(blur - i, blur - i));
			}else {
				inMat.copyTo(out);
			}
			//Size size = new Size(in.getWidth() * (0.9 + (i/(540/0.1))), in.getHeight() * (0.9 + (i/(540/0.1))));
			//Mat cropped = out.submat(new Rect((int)(in.getWidth()-size.width), (int)(in.getHeight()-size.height), (int)size.width, (int)size.height));
			//Imgproc.resize(cropped, cropped, new Size(3840, 2160));
			BufferedImage outBi = CoreSystems.toBufferedImage(HighGui.toBufferedImage(out));
			File file = new File("raw/2/out/" + i + ".jpg");
			ImageIO.write(outBi, "jpg", file);
		}
	}

}
