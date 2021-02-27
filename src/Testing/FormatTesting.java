package Testing;

import java.awt.BorderLayout;
import java.awt.event.WindowAdapter;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import WebcamTests.WebcamDrawer;
import WebcamTests.WebcamTester;

public class FormatTesting extends WindowAdapter{
	
	private static FormatTesting instance = new FormatTesting();
	
	public static void main(String[] args) throws IOException {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		instance.start();
	}
	
	public void start() throws IOException {
		BufferedImage navTarget = ImageIO.read(new File("raw/6TToXXph.jpg"));
		Mat navMat = bufferedImageToMat(navTarget);
		Imgproc.cvtColor(navMat, navMat, Imgproc.COLOR_BGR2HSV);
		//Scalar min = new Scalar(110,56,56); //0, 49.1, 43.1
		//Scalar max = new Scalar(200,121,121); //0 39.5 78.4
		
		//Scalar min = new Scalar(0, 100, 110);
		//Scalar max = new Scalar(0, 125, 200);
		
		//Scalar min = new Scalar(0, 80, 90);
		//Scalar max = new Scalar(360, 255, 230);
		
		WebcamDrawer drawer = new WebcamDrawer(new Size(1280, 720));
		
		JFrame frame = new JFrame("Webcam");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		frame.getContentPane().add(drawer, BorderLayout.CENTER);
		
		JSlider H = new JSlider(JSlider.HORIZONTAL, 0, 180, 6);
		JSlider S = new JSlider(JSlider.HORIZONTAL, 0, 255, 151);
		JSlider V = new JSlider(JSlider.HORIZONTAL, 0, 255, 99);//126 - 121
		
		H.setPaintLabels(true);
		S.setPaintLabels(true);
		V.setPaintLabels(true);
		
		JPanel panel = new JPanel();
		
		panel.add(H, BorderLayout.EAST);
		panel.add(S, BorderLayout.CENTER);
		panel.add(V, BorderLayout.WEST);
		
		frame.add(panel, BorderLayout.NORTH);
		
		JSlider H2 = new JSlider(JSlider.HORIZONTAL, 0, 180, 23);
		JSlider S2 = new JSlider(JSlider.HORIZONTAL, 0, 255, 255);
		JSlider V2 = new JSlider(JSlider.HORIZONTAL, 0, 255, 255);
		
		H2.setPaintLabels(true);
		S2.setPaintLabels(true);
		V2.setPaintLabels(true);
		
		JPanel panel2 = new JPanel();
		
		panel2.add(H2, BorderLayout.EAST);
		panel2.add(S2, BorderLayout.CENTER);
		panel2.add(V2, BorderLayout.WEST);
		
		frame.add(panel2, BorderLayout.SOUTH);
		
		frame.repaint();
		frame.setSize(1280, 850);
		frame.setVisible(true);
		
		while(true) {
			Scalar min = new Scalar(H.getValue(), S.getValue(), V.getValue());
			Scalar max = new Scalar(H2.getValue(), S2.getValue(), V2.getValue());
			
			Mat mask = new Mat();
			Core.inRange(navMat, min, max, mask);
			drawer.setMat(mask);
			frame.repaint();
			System.out.println(H.getValue() + ", " + S.getValue() + ", " + V.getValue() + " | " + H2.getValue() + ", " + S2.getValue() + ", " + V2.getValue());
		}
		
	}
	
	public static Mat bufferedImageToMat(BufferedImage bi) {
		  Mat mat = new Mat(bi.getHeight(), bi.getWidth(), CvType.CV_8UC3);
		  byte[] data = ((DataBufferByte) bi.getRaster().getDataBuffer()).getData();
		  mat.put(0, 0, data);
		  return mat;
	}
}
