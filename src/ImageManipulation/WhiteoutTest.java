package ImageManipulation;

import java.awt.BorderLayout;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.JFrame;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import WebcamTests.WebcamDrawer;

public class WhiteoutTest {

	public static void main(String[] args) throws IOException {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		BufferedImage in = ImageIO.read(new File("res/ImageManipulation/Test4.jpg"));
		Mat inMat = bufferedImageToMat(in);
		Imgproc.resize(inMat, inMat, new Size(in.getWidth()/4, in.getHeight()/4));
		Mat rgb = new Mat();
		Imgproc.cvtColor(inMat, rgb, Imgproc.COLOR_BGR2RGB);
		
		
		WebcamDrawer drawer = new WebcamDrawer(new Size(1280, 720));
		
		JFrame frame = new JFrame("Webcam");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		frame.getContentPane().add(drawer, BorderLayout.CENTER);
		
		frame.repaint();
		frame.setSize(700, 600);
		frame.setVisible(true);
		
		while(true) {
			Scalar min = new Scalar(170, 0, 0);
			Scalar max = new Scalar(255, 140, 140);
			Mat outMat = new Mat();
			Core.inRange(rgb, min, max, outMat);
			Mat testMat = new Mat();
			
			drawer.setMat(outMat);
			frame.repaint();
		}
	}
	
	public static Mat bufferedImageToMat(BufferedImage bi) {
		  Mat mat = new Mat(bi.getHeight(), bi.getWidth(), CvType.CV_8UC3);
		  byte[] data = ((DataBufferByte) bi.getRaster().getDataBuffer()).getData();
		  mat.put(0, 0, data);
		  return mat;
	}

}
