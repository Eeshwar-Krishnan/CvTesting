package PnP;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
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

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.BFMatcher;
import org.opencv.features2d.BRISK;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.Features2d;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

import com.google.gson.Gson;

public class Runner {
	public static void main(String[] args) throws IOException {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		BufferedImage calibImage = ImageIO.read(new File("res/PnP/calibration.jpg"));
		Mat calibMat = bufferedImageToMat(calibImage);
		
		List<Mat> objectPointsArr = new ArrayList<>();
		List<Mat> imagePointsArr = new ArrayList<>();
		
		File imgDir = new File("res/PnP/imgs");
		File[] inImgs = imgDir.listFiles();
		
		for(int i = 0; i < 15; i ++) {
			File f = inImgs[i];
			//addTrain(imagePointsArr, objectPointsArr, f);
			System.out.println("Finished Training " + (i + 1) + " out of " + 15);
		}
		System.out.println();
		Mat cameraMatrix = new Mat();
		MatOfDouble distCoeffs = new MatOfDouble();
		Mat intrinsic = new Mat(3, 3, CvType.CV_32FC1);
		intrinsic.put(0, 0, 1);
		intrinsic.put(1, 1, 1);
		
		List<Mat> rvecs = new ArrayList<>();
		List<Mat> tvecs = new ArrayList<>();
		
		Mat objectsPoints = new Mat();
						
		//Calib3d.calibrateCamera(objectPointsArr, imagePointsArr, new Size(11, 8.5), intrinsic, distCoeffs, rvecs, tvecs);
	
		double[] inrinsicFloat = new double[] {817.063304531327,0.0,325.9485286458284,0.0,819.4690054531818,236.2597899599986,0.0,0.0,1.0,0.0};
		intrinsic.put(0, 0, inrinsicFloat);
		
		System.out.println();
		
		double[] distFloat = new double[] {-0.014680796227423968,1.3720322590501144,-0.0028429009326778093,0.0010064951672061734,-5.347658630748131};
		distCoeffs.fromArray(distFloat);
		
		Mat rvec = new Mat();
		Mat tvec = new Mat();
		
		BufferedImage navTarget = ImageIO.read(new File("res/NavTargets/redalliancewalltarget.jpg"));
		
		Mat navMat = bufferedImageToMat(navTarget);
		Imgproc.resize(navMat, navMat, new Size(navTarget.getWidth()/5, navTarget.getHeight()/5));
		
		solvePnP(bufferedImageToMat(ImageIO.read(new File("res/RealTests/test1.jpg"))), navMat, intrinsic, distCoeffs, rvec, tvec, navTarget.getWidth()/5, navTarget.getHeight()/5);
		
		System.out.println(rvec.dump());
		System.out.println(tvec.dump());
		
		Mat rod = new Mat();
		
		Calib3d.Rodrigues(rvec, rod);
		
		Mat transpose = new Mat();
		
		Core.transpose(rod, transpose);
		
		Mat inversed = new Mat();
		
		Core.scaleAdd(transpose, -1, Mat.zeros(transpose.size(), transpose.type()), inversed);
		
		Mat multiplyResult = new Mat();
		
		System.out.println();
		
		System.out.println(tvec.dump());
		System.out.println(inversed.dump());
		
		Core.gemm(inversed, tvec, 1, new Mat(), 0, multiplyResult);
		
		System.out.println(multiplyResult.dump());
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
	
	public static void addTrain(List<Mat> vec2fs, List<Mat> vec3fs, File toTrain) throws IOException {
		BufferedImage testImage = ImageIO.read(toTrain);
		Mat testMat = bufferedImageToMat(testImage);
		
		Gson gson = new Gson();
		File f = new File("res/PnP/classifier.conf");
		BufferedReader reader = new BufferedReader(new FileReader(f));
		String s = reader.readLine(), fullText = "";
		while(s != null) {
			fullText += s;
			s = reader.readLine();
		}
		
		ClassifierFormat format = gson.fromJson(fullText, ClassifierFormat.class);
		
		MatOfKeyPoint keypointMat = new MatOfKeyPoint();
		keypointMat.create(format.keyrows, format.keycols, format.keytype);
		keypointMat.put(0, 0, format.keydata);
		
		Mat keypointDesc = new Mat();
		keypointDesc.create(format.descrows, format.desccols, format.desctype);
		keypointDesc.put(0, 0, format.descdata);
		
		MatOfKeyPoint testKeypoint = new MatOfKeyPoint();
		Mat testDesc = new Mat();
		
		BRISK classifier = BRISK.create();
		classifier.detectAndCompute(testMat, testMat, testKeypoint, testDesc);
		
		List<MatOfDMatch> matches = new ArrayList<>();
		
		BFMatcher matcher = BFMatcher.create();
		matcher.knnMatch(keypointDesc, testDesc, matches, 2);
				
		ArrayList<DMatch> arr = new ArrayList<>();
		for(MatOfDMatch mD : matches) {
			if(mD.toArray()[0].distance < (0.8 * mD.toArray()[1].distance)) {
				arr.add(mD.toArray()[0]);
				arr.add(mD.toArray()[1]);
			}
			
		}
		
		Collections.sort(arr, new Comparator<DMatch>() {

			@Override
			public int compare(DMatch o1, DMatch o2) {
				return (int)(o1.distance - o2.distance);
			}
			
		});
		MatOfDMatch match2 = new MatOfDMatch(arr.subList(0, (int)(arr.size() * 0.25)).toArray(new DMatch[10]));
		
		ArrayList<DMatch> computedMatches = new ArrayList<>();
		computedMatches.addAll(arr.subList(0, (int)(arr.size())));
		
		MatOfPoint2f point2fs = new MatOfPoint2f();
		MatOfPoint3f point3fs = new MatOfPoint3f();
		double maxX = 0;
		double maxY = 0;
		for(DMatch match : computedMatches) {
			KeyPoint kp = keypointMat.toArray()[match.queryIdx];
			Point3 p = new Point3((kp.pt.x / (double)format.imgWidth) * (11), (kp.pt.y / (double)format.imgHeight) * (8.5), 0);
			point3fs.push_back(new MatOfPoint3f(p));
			point2fs.push_back(new MatOfPoint2f(testKeypoint.toArray()[match.trainIdx].pt));
			maxX = Math.max(p.x, maxX);
			maxY = Math.max(p.y, maxY);
		}
		//System.out.println(maxX + " | " + maxY);
		vec2fs.add(point2fs);
		vec3fs.add(point3fs);
	}
	
	private static boolean solvePnP(Mat toSolve, Mat calibImage, Mat cameraMatrix, MatOfDouble distCoeffs, Mat rvec, Mat tvec, int width, int height) throws IOException {
		Mat testMat = toSolve.clone();
		Imgproc.cvtColor(testMat, testMat, Imgproc.COLOR_BGR2GRAY);
		
		MatOfKeyPoint testKeypoint = new MatOfKeyPoint();
		Mat testDesc = new Mat();

		BRISK classifier = BRISK.create();
		
		classifier.detect(testMat, testKeypoint);
				
		classifier.compute(testMat, testKeypoint, testDesc);
		
		MatOfKeyPoint calibKeypoint = new MatOfKeyPoint();
		Mat calibDesc = new Mat();
		
		classifier.detect(calibImage, calibKeypoint);
				
		classifier.compute(calibImage, calibKeypoint, calibDesc);
		
		//instance.classifier.detectAndCompute(testMat, testMat, testKeypoint, testDesc);
		
		List<MatOfDMatch> matches = new ArrayList<>();
		
		DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
		matcher.knnMatch(calibDesc, testDesc, matches, 2);
		
		ArrayList<DMatch> arr = new ArrayList<>();
		for(MatOfDMatch mD : matches) {
			if(mD.toArray().length > 1) {
				if(mD.toArray()[0].distance < (0.6 * mD.toArray()[1].distance)) {
					arr.add(mD.toArray()[0]);
				}
			}
		}
		
		/*for(MatOfDMatch mD : matches) {
			if(mD.toArray().length > 0)
				arr.add(mD.toArray()[0]);
			//System.out.println(mD.toArray().length);
		}
		*/
		Collections.sort(arr, new Comparator<DMatch>() {

			@Override
			public int compare(DMatch o1, DMatch o2) {
				return (int)(o1.distance - o2.distance);
			}
			
		});
		Mat outImg = new Mat();
		Features2d.drawKeypoints(toSolve, testKeypoint, outImg);
		//instance.drawer.setMat(outImg);
		if(arr.size() > 5) {
			try {
				List<DMatch> betterMatches = new ArrayList<>();
				KeyPoint[] arr1 = calibKeypoint.toArray();
				KeyPoint[] arr2 = testKeypoint.toArray();
				MatOfPoint2f pts1Mat = new MatOfPoint2f();
				MatOfPoint2f pts2Mat = new MatOfPoint2f();
				for(int i = 0; i < arr.size(); i ++) {
					pts1Mat.push_back(new MatOfPoint2f(arr1[arr.get(i).queryIdx].pt));
					pts2Mat.push_back(new MatOfPoint2f(arr2[arr.get(i).trainIdx].pt));
				}
				Mat mask = new Mat();
				Calib3d.findHomography(pts1Mat, pts2Mat, Calib3d.RANSAC, 15, mask);
				
				for(int i = (arr.size()-1); i >= 0; i --) {
					if(mask.get(i, 0)[0] != 0.0) {
						betterMatches.add(arr.get(i));
					}
				}
				if(betterMatches.size() > 5) {
					arr.clear();
					arr.addAll(betterMatches);
				}
				/*MatOfDMatch match2;
				if(arr.size() > 30) {
					//match2 = new MatOfDMatch(arr.subList(0, (int)(arr.size()*0.25)).toArray(new DMatch[10]));
					match2 = new MatOfDMatch(arr.subList(0, (int)(arr.size())).toArray(new DMatch[10]));
				}else {
					match2 = new MatOfDMatch(arr.subList(0, (int)(arr.size()*0.15)).toArray(new DMatch[10]));
				}
				*/
				//MatOfDMatch match2 = new MatOfDMatch(arr.toArray(new DMatch[10]));
				//Features2d.drawMatches(calibImage, instance.keypoints, toSolve, testKeypoint, match2, outImg);
				//instance.drawer.setMat(outImg);
				//Calib3d.undistort(toSolve, outImg, cameraMatrix, distCoeffs);
				//instance.drawer.setMat(outImg);
				
				
				
				ArrayList<DMatch> computedMatches = new ArrayList<>();
				if(arr.size() > 20) {
					System.out.println(arr.size());
					computedMatches.addAll(arr.subList(0, (int)(20)));
				}else {
					System.out.println("Inconsistency Detected!");
					computedMatches.addAll(arr);
				}
				
				MatOfPoint2f point2fs = new MatOfPoint2f();
				MatOfPoint3f point3fs = new MatOfPoint3f();
				for(DMatch match : computedMatches) {
					KeyPoint kp = calibKeypoint.toArray()[match.queryIdx];
					Point3 p = new Point3(((kp.pt.x / (double)width) * (11)) - (11.0/2.0), ((kp.pt.y / (double)height) * (8.5)) - (8.5/2.0), 0);
					point3fs.push_back(new MatOfPoint3f(p));
					point2fs.push_back(new MatOfPoint2f(testKeypoint.toArray()[match.trainIdx].pt));
				}
				
				//Calib3d.solvePnP(point3fs, point2fs, cameraMatrix, distCoeffs, rvec, tvec);
				Calib3d.solvePnPRansac(point3fs, point2fs, cameraMatrix, distCoeffs, rvec, tvec);
				//Calib3d.solvePnP(point3fs, point2fs, cameraMatrix, distCoeffs, rvec, tvec);
				//System.out.println((timer - System.currentTimeMillis()) + " | " + (timer2 - System.currentTimeMillis()) + " | " + (timer3 - System.currentTimeMillis()) + " | " + (timer4 - System.currentTimeMillis()) + " | " + (timer5 - System.currentTimeMillis()) + " | " + (timer6 - System.currentTimeMillis()));

				return true;
			}catch(NullPointerException e) {
				return false;
			}
		}else {
			return false;
		}
	}
	
	
}
