package ImageManipulation;

import java.awt.BorderLayout;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.JFrame;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import AnimTests.CoreSystems;

public class TowerGoalWebcam {
	public static void main(String[] args) throws IOException {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		BufferedImage inm = ImageIO.read(new File("res/ImageManipulation/Match/Capture.jpg"));
		Mat inMatM = CoreSystems.bufferedImageToMat(inm);
		Imgproc.cvtColor(inMatM, inMatM, Imgproc.COLOR_BGR2GRAY);
		Imgproc.threshold(inMatM, inMatM, 1, 255, Imgproc.THRESH_BINARY);
		List<MatOfPoint> matchContours = new ArrayList<>();
		Imgproc.findContours(inMatM, matchContours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);
		double maxVal = 0;
		int maxValIdx = 0;
		for (int contourIdx = 0; contourIdx < matchContours.size(); contourIdx++)
		{
		    double contourArea = Imgproc.contourArea(matchContours.get(contourIdx));
		    if (maxVal < contourArea)
		    {
		        maxVal = contourArea;
		        maxValIdx = contourIdx;
		    }
		}
		MatOfPoint match = matchContours.get(maxValIdx);
		MatOfPoint2f matchCurve = new MatOfPoint2f();
		Imgproc.approxPolyDP(new MatOfPoint2f(match.toArray()), matchCurve, 0.01 * Imgproc.arcLength(new MatOfPoint2f(match.toArray()), true), true);
		
		JFrame frame = new JFrame("Webcam");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		MultiMatDrawer drawer = new MultiMatDrawer();
		
		drawer.setOne(CoreSystems.bufferedImageToMat(ImageIO.read(new File("res/RealTests/Test1.jpg"))));
		drawer.setTwo(CoreSystems.bufferedImageToMat(ImageIO.read(new File("res/RealTests/Test2.jpg"))));
		drawer.setThree(CoreSystems.bufferedImageToMat(ImageIO.read(new File("res/RealTests/Test3.jpg"))));
		drawer.setFour(CoreSystems.bufferedImageToMat(ImageIO.read(new File("res/RealTests/Test4.jpg"))));
				
		frame.getContentPane().add(drawer, BorderLayout.CENTER);
				
		frame.repaint();
		frame.setSize(1280, 720);
		frame.setVisible(true);
		
		VideoCapture capture = new VideoCapture(0);
		while(!capture.isOpened());
		long lastTime = System.currentTimeMillis();
		while(true) {
			drawer.setFps(1.0/((System.currentTimeMillis() - lastTime)/1000.0));
			lastTime = System.currentTimeMillis();
			Mat raw = new Mat();
			capture.read(raw);
			drawer.setOne(raw);
			
			Mat rgb = new Mat();
			Imgproc.cvtColor(raw, rgb, Imgproc.COLOR_BGR2RGB);
			Scalar min = new Scalar(215, 0, 0);
			Scalar max = new Scalar(255, 120, 120);
			Mat outMat = new Mat();
			Core.inRange(rgb, min, max, outMat);
			Mat testMat = new Mat();
			raw.copyTo(testMat, outMat);
						
			Imgproc.cvtColor(testMat, testMat, Imgproc.COLOR_RGB2GRAY);
			Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(10, 10));
			Imgproc.morphologyEx(testMat, testMat, Imgproc.MORPH_CLOSE, kernel);
			
			Imgproc.threshold(testMat, testMat, 1, 255, Imgproc.THRESH_BINARY);
			
			drawer.setTwo(testMat.clone());
			
			List<MatOfPoint> contours = new ArrayList<>();
			Imgproc.findContours(testMat, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
			
			double minMatch = 100;
			MatOfPoint bestCont = new MatOfPoint();
			int ind = 0;
			for(int i = 0; i < contours.size(); i ++) {
				MatOfPoint m = contours.get(i);
				MatOfPoint2f mat2f = new MatOfPoint2f(m.toArray());
				double d = Imgproc.arcLength(mat2f, true);
				if(d > 500) {
					double matchVal = Imgproc.matchShapes(m, match, 1, 0);
					if(matchVal < 0.3 || true) {
						if(matchVal < minMatch) {
							bestCont = m;
							minMatch = matchVal;
							ind = i;
						}
					}
				}
			}
						
			if(minMatch != 100) {
				Mat drawer3 = Mat.zeros(testMat.size(), CvType.CV_8UC3);
				MatOfPoint m = bestCont;
				MatOfPoint2f mat2f = new MatOfPoint2f(m.toArray());
				
				Rect rect = Imgproc.boundingRect(mat2f);
				Mat mask = Mat.zeros(testMat.size(), CvType.CV_8U);;
				for(int i = rect.x; i < (rect.x + rect.width); i ++) {
					for(int j = rect.y; j < (rect.y + rect.height); j ++) {
						mask.put(j, i, 255);
					}
				}
				Mat out = Mat.zeros(testMat.size(), CvType.CV_8U);
				testMat.copyTo(out, mask);
				Imgproc.drawContours(drawer3, contours, ind, new Scalar(255, 0, 0), 2);
				Imgproc.rectangle(drawer3, rect, new Scalar(0, 255, 0), 2);
				drawer.setThree(drawer3);
				MatOfPoint2f approxCurve = new MatOfPoint2f();
				Imgproc.approxPolyDP(mat2f, approxCurve, 0.01 * Imgproc.arcLength(mat2f, true), true);
			}
			frame.repaint();
		}
	}
}
