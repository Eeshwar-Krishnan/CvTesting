package PnP;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.JButton;
import javax.swing.JFrame;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.BFMatcher;
import org.opencv.features2d.BRISK;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.ORB;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import com.google.gson.Gson;

import WebcamTests.WebcamDrawer;
import WebcamTests.WebcamTester;

public class WebcamCalibration extends WindowAdapter{
	
	private boolean finished;
	
	private static WebcamCalibration instance = new WebcamCalibration();
	
	public Mat cameraMatrix;
	public MatOfDouble distCoeffs;
	public WebcamDrawer drawer, drawer2;
	public ClassifierFormat format;
	public MatOfKeyPoint keypoints;
	public Mat descriptors;
	public BRISK classifier, trainClassifier;
	
	public static void main(String[] args) throws IOException {
		
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		instance.start();
	}
	
	public void start() throws IOException{
		WebcamDrawer drawer = new WebcamDrawer(new Size(1280, 720));
		this.drawer = drawer;
		
		JFrame frame = new JFrame("Webcam");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		frame.getContentPane().add(drawer, BorderLayout.CENTER);
		JButton captureButton = new JButton("Capture");
		frame.getContentPane().add(captureButton, BorderLayout.SOUTH);
		JButton run = new JButton("Run");
		frame.getContentPane().add(run, BorderLayout.NORTH);
		
		frame.repaint();
		frame.setSize(700, 600);
		frame.setVisible(true);
		
		MatOfPoint3f obj = new MatOfPoint3f();
		
		for(int i = 0; i < (6); i ++) {
			for(int j = 0; j < 9; j ++) {
				obj.push_back(new MatOfPoint3f(new Point3((j), (i), 0)));
			}
		}
		
		System.out.println(obj.dump());
		
		ArrayList<Mat> imgPoints = new ArrayList<>();
		ArrayList<Mat> objPoints = new ArrayList<>();
		
		VideoCapture capture = new VideoCapture(700);
		Mat image = new Mat();
		boolean prevCaptureState = false, prevRunState = false;
		int numCaptured = 0;
		if(capture.isOpened()) {
			long lastTime = System.currentTimeMillis();
			while(!finished) {
				Mat img = new Mat();
				capture.read(img);
				Mat grayScale = new Mat();
				Imgproc.cvtColor(img, grayScale, Imgproc.COLOR_BGRA2GRAY);
				Size size = new Size(9, 6);
				MatOfPoint2f corners = new MatOfPoint2f();
				boolean found = Calib3d.findChessboardCorners(grayScale, size, corners);
				Mat grayCopy = new Mat();
				grayScale.copyTo(grayCopy);
				if(found) {
					TermCriteria term = new TermCriteria(TermCriteria.EPS | TermCriteria.MAX_ITER, 30, 0.1);
					Imgproc.cornerSubPix(grayScale, corners, new Size(11, 11), new Size(-1, -1), term);
					Calib3d.drawChessboardCorners(img, size, corners, found);
					//System.out.println("Found");
					if(captureButton.getModel().isPressed() && (captureButton.getModel().isPressed() != prevCaptureState)) {
						imgPoints.add(corners);
						objPoints.add(obj);
						System.out.println("Captured " + (numCaptured + 1) + " of 15");
						numCaptured ++;
					}
				}
				prevCaptureState = captureButton.getModel().isPressed();
				if(run.getModel().isPressed() && (run.getModel().isPressed() != prevRunState)) {
					List<Mat> rvecs = new ArrayList<>();
					List<Mat> tvecs = new ArrayList<>();
					Mat intrinsic = new Mat(3, 3, CvType.CV_32FC1);
					intrinsic.put(0, 0, 1);
					intrinsic.put(1, 1, 1);
					Mat distCoeffs = new Mat();
					Calib3d.calibrateCamera(objPoints, imgPoints, grayCopy.size(), intrinsic, distCoeffs, rvecs, tvecs);
					System.out.println();
					double[] intrinsicData = new double[(int)intrinsic.total() + intrinsic.channels()];
					double[] distCoeffsData = new double[(int)distCoeffs.total() + distCoeffs.channels()];
					intrinsic.get(0, 0, intrinsicData);
					distCoeffs.get(0, 0, distCoeffsData);
					for(double d : intrinsicData) {
						System.out.print(d + ",");
					}
					System.out.println();
					for(double d : distCoeffsData) {
						System.out.print(d + ",");
					}
				}
				prevRunState = run.getModel().isPressed();
				drawer.setMat(img);
				frame.repaint();
			}
		}
	}
	
	private WebcamCalibration() {
		finished = false;
	}
	
    @Override
    public void windowClosed(WindowEvent e) {
    	finished = true;
    }
	
	public static Mat bufferedImageToMat(BufferedImage bi) {
		  Mat mat = new Mat(bi.getHeight(), bi.getWidth(), CvType.CV_8UC3);
		  byte[] data = ((DataBufferByte) bi.getRaster().getDataBuffer()).getData();
		  mat.put(0, 0, data);
		  return mat;
	}
	
	public static BufferedImage toBufferedImage(Image img)
	{
	    if (img instanceof BufferedImage)
	    {
	        return (BufferedImage) img;
	    }

	    // Create a buffered image with transparency
	    BufferedImage bimage = new BufferedImage(img.getWidth(null), img.getHeight(null), BufferedImage.TYPE_INT_ARGB);

	    // Draw the image on to the buffered image
	    Graphics2D bGr = bimage.createGraphics();
	    bGr.drawImage(img, 0, 0, null);
	    bGr.dispose();

	    // Return the buffered image
	    return bimage;
	}
	
	public static double cmToIn(double cm) {
		return cm/2.54;
	}
	
	
}
