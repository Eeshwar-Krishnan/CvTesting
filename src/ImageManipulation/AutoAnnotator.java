package ImageManipulation;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.Features2d;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

public class AutoAnnotator {

	public static void main(String[] args) throws IOException {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		BufferedImage in = ImageIO.read(new File("res/ImageManipulation/Test4.jpg"));
		Mat inMat = bufferedImageToMat(in);
		BufferedImage inm = ImageIO.read(new File("res/ImageManipulation/Match/Capture.jpg"));
		Mat inMatM = bufferedImageToMat(inm);
		//Imgproc.resize(inMatM, inMatM, new Size(inm.getWidth()/4, inm.getHeight()/4));
		long timerStart = System.nanoTime();
		Imgproc.cvtColor(inMatM, inMatM, Imgproc.COLOR_BGR2GRAY);
		Mat rgb = new Mat();
		Imgproc.resize(inMat, inMat, new Size(in.getWidth()/4, in.getHeight()/4));
		Imgproc.cvtColor(inMat, rgb, Imgproc.COLOR_BGR2RGB);
		Scalar min = new Scalar(210, 0, 0);
		Scalar max = new Scalar(255, 120, 120);
		Mat outMat = new Mat();
		Core.inRange(rgb, min, max, outMat);
		Mat testMat = new Mat();
		inMat.copyTo(testMat, outMat);
				
		Imgproc.cvtColor(testMat, testMat, Imgproc.COLOR_RGB2GRAY);
		Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(10, 10));
		Mat kernel2 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2, 2));
		//Imgproc.morphologyEx(testMat, testMat, Imgproc.MORPH_ERODE, kernel2);
		Imgproc.morphologyEx(testMat, testMat, Imgproc.MORPH_CLOSE, kernel);
		
		Imgproc.threshold(testMat, testMat, 1, 255, Imgproc.THRESH_BINARY);
		Imgproc.threshold(inMatM, inMatM, 1, 255, Imgproc.THRESH_BINARY);

		List<MatOfPoint> contours = new ArrayList<>();
		Imgproc.findContours(testMat, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
		
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
		double minMatch = 100;
		int ind = 0;
		MatOfPoint bestCont = new MatOfPoint();
		for(int i = 0; i < contours.size(); i ++) {
			MatOfPoint m = contours.get(i);
			MatOfPoint2f mat2f = new MatOfPoint2f(m.toArray());
			double d = Imgproc.arcLength(mat2f, true);
			if(d > 500) {
				double matchVal = Imgproc.matchShapes(m, match, 1, 0);
				if(matchVal < 1) {
					if(matchVal < minMatch) {
						bestCont = m;
						minMatch = matchVal;
						ind = i;
					}
				}
			}
		}
		System.out.println("e " + minMatch);
		if(minMatch != 100) {
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
			
			RotatedRect rotRect = Imgproc.minAreaRect(mat2f);
			double angle = rotRect.angle;
			if(rotRect.size.width < rotRect.size.height) {
				angle += 90;
			}
			System.out.println(angle);
			Mat matrix = Imgproc.getRotationMatrix2D(new Point(out.size().width/2, out.size().height/2), angle, 1);
			Mat rotated = new Mat();
			Imgproc.warpAffine(out, rotated, matrix, out.size());
			//out.copyTo(rotated);
			
			matchContours = new ArrayList<>();
			
			Imgproc.findContours(rotated, matchContours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
			
			System.out.println(matchContours.size());
						
			mat2f = new MatOfPoint2f(matchContours.get(0).toArray());
			
			double d = Imgproc.arcLength(mat2f, true);
			MatOfPoint2f approxCurve = new MatOfPoint2f();
			Imgproc.approxPolyDP(mat2f, approxCurve, 0.005 * d, false);
			rect = Imgproc.boundingRect(mat2f);
			
			Mat fakedTrain = Mat.zeros(testMat.rows(), testMat.cols(), testMat.type());
			MatOfPoint2f point2fs = new MatOfPoint2f();
			for(Point p : approxCurve.toArray()) {
				Imgproc.circle(fakedTrain, p, 2, new Scalar(255, 255, 255), -1);
				point2fs.push_back(new MatOfPoint2f(new Point(p.x - rect.x, p.y - rect.y)));
			}
			
			Imgproc.resize(inMatM, inMatM, new Size(rect.width, rect.height));
			
			matchContours = new ArrayList<>();
			Imgproc.findContours(inMatM, matchContours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);
			
			MatOfPoint2f matchCurve = new MatOfPoint2f();
			Imgproc.approxPolyDP(new MatOfPoint2f(matchContours.get(0).toArray()), matchCurve, 0.005 * Imgproc.arcLength(new MatOfPoint2f(matchContours.get(0).toArray()), true), true);
			Rect matchRect = Imgproc.boundingRect(matchCurve);
			
			double epsilon = 15;
			MatOfDMatch dmatches = new MatOfDMatch();
			MatOfKeyPoint m1 = new MatOfKeyPoint();
			MatOfKeyPoint m2 = new MatOfKeyPoint();
			MatOfPoint3f point3fs = new MatOfPoint3f();
			MatOfPoint2f truePoint2fs = new MatOfPoint2f();
			for(Point p : matchCurve.toArray()) {
				for(Point pt : point2fs.toArray()) {
					if(Math.abs(p.x - pt.x) < epsilon && Math.abs(p.y - pt.y) < epsilon) {
						m1.push_back(new MatOfKeyPoint(new KeyPoint((float)pt.x+rect.x, (float)pt.y+rect.y, -1)));
						m2.push_back(new MatOfKeyPoint(new KeyPoint((float)p.x, (float)p.y, -1)));
						dmatches.push_back(new MatOfDMatch(new DMatch(m1.toArray().length-1, m2.toArray().length-1, 1)));
						point3fs.push_back(new MatOfPoint3f(new Point3((float)((p.x/(matchRect.width + matchRect.x)) * (23.875)), (float)((p.y/(matchRect.height + matchRect.y)) * (15.75)), 0)));
						truePoint2fs.push_back(new MatOfPoint2f(pt));
						System.out.println(point3fs.toArray()[point3fs.toArray().length-1] + " | " + truePoint2fs.toArray()[truePoint2fs.toArray().length-1]);
						//System.out.println(p.x + ", " + (matchRect.width + matchRect.x));
					}
				}
			}
			
			d = Imgproc.arcLength(new MatOfPoint2f(match.toArray()), true);
			approxCurve = new MatOfPoint2f();
			Imgproc.approxPolyDP(new MatOfPoint2f(match.toArray()), approxCurve, 0.01 * d, true);
			Mat fakedQuery = Mat.zeros(inMatM.size(), inMatM.type());
			for(Point p : approxCurve.toArray()) {
				Imgproc.circle(fakedQuery, p, 2, new Scalar(255, 255, 255), -1);
			}
			
			Mat bruh = new Mat();
			Features2d.drawMatches(rotated, m1, inMatM, m2, dmatches, bruh);
			
			MatOfDouble distCoeffs = new MatOfDouble();
			Mat intrinsic = new Mat(3, 3, CvType.CV_32FC1);
			Mat rvec = new Mat();
			Mat tvec = new Mat();
		
			double[] inrinsicFloat = new double[] {817.063304531327,0.0,325.9485286458284,0.0,819.4690054531818,236.2597899599986,0.0,0.0,1.0,0.0};
			intrinsic.put(0, 0, inrinsicFloat);
						
			double[] distFloat = new double[] {-0.014680796227423968,1.3720322590501144,-0.0028429009326778093,0.0010064951672061734,-5.347658630748131};
			distCoeffs.fromArray(distFloat);
			
			System.out.println(point3fs.toArray().length + ", " + truePoint2fs.toArray().length);
			
			boolean test = Calib3d.solvePnPRansac(point3fs, truePoint2fs, intrinsic, distCoeffs, rvec, tvec);
			System.out.println(test);
			
			System.out.println(toPosition(tvec, rvec).dump());
			
			System.out.println((System.nanoTime() - timerStart) / 1_000_000_000.0 + " seconds");
			
			HighGui.imshow("Test", bruh);
			HighGui.waitKey();
		}else {
			System.out.println("No Match Found");
		}
	}
	
	public static Mat bufferedImageToMat(BufferedImage bi) {
		  Mat mat = new Mat(bi.getHeight(), bi.getWidth(), CvType.CV_8UC3);
		  byte[] data = ((DataBufferByte) bi.getRaster().getDataBuffer()).getData();
		  mat.put(0, 0, data);
		  return mat;
	}
	
	public static Mat toPosition(Mat tvec, Mat rvec) {
		Mat rod = new Mat();
		
		Calib3d.Rodrigues(rvec, rod);
		
		Mat transpose = new Mat();
		
		Core.transpose(rod, transpose);
		
		Mat inversed = new Mat();
		
		Core.scaleAdd(transpose, -1, Mat.zeros(transpose.size(), transpose.type()), inversed);
		
		Mat multiplyResult = new Mat();
		
		Core.gemm(inversed, tvec, 1, new Mat(), 0, multiplyResult);
		
		return multiplyResult;
	}
}
