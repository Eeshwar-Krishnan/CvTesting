package Testing;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import javax.imageio.ImageIO;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

public class ImageTesting {

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		//6, 151, 126 | 23, 255, 255
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		BufferedImage navTarget = ImageIO.read(new File("raw/am-4300.jpg"));
		Mat navMat = bufferedImageToMat(navTarget);
		Imgproc.cvtColor(navMat, navMat, Imgproc.COLOR_BGR2HSV);
		
		Scalar min = new Scalar(6, 151, 99);
		Scalar max = new Scalar(23, 255, 255);
		
		Mat mask = new Mat();
		Core.inRange(navMat, min, max, mask);
		
		ArrayList<MatOfPoint> contours = new ArrayList<>();
		
		Imgproc.findContours(mask, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);
		
		int largestIndex = 0;
		double maxSize = 0;
		
		for(int i = 0; i < contours.size(); i ++) {
			double area = Imgproc.contourArea(contours.get(i));
			if(maxSize < area) {
				maxSize = area;
				largestIndex = i;
			}
		}
		
		MatOfPoint2f matchCurve = new MatOfPoint2f();
		Imgproc.approxPolyDP(new MatOfPoint2f(contours.get(largestIndex).toArray()), matchCurve, 0.005 * Imgproc.arcLength(new MatOfPoint2f(contours.get(largestIndex).toArray()), true), true);
		
		Rect matchRect = Imgproc.boundingRect(matchCurve);
		
		Imgproc.rectangle(mask, matchRect, new Scalar(255, 0, 0));
		
		Mat test = new Mat();
		Core.copyTo(mask, test, new Mat());
		
		HighGui.imshow("Test", test);
		HighGui.waitKey();
	}
	
	public static Mat bufferedImageToMat(BufferedImage bi) {
		  Mat mat = new Mat(bi.getHeight(), bi.getWidth(), CvType.CV_8UC3);
		  byte[] data = ((DataBufferByte) bi.getRaster().getDataBuffer()).getData();
		  mat.put(0, 0, data);
		  return mat;
	}

}
