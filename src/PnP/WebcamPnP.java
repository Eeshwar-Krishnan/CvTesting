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
import javax.swing.JFrame;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.BFMatcher;
import org.opencv.features2d.BRISK;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FastFeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.ORB;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import com.google.gson.Gson;

import WebcamTests.WebcamDrawer;
import WebcamTests.WebcamTester;

public class WebcamPnP extends WindowAdapter{
	
	private boolean finished;
	
	private static WebcamPnP instance = new WebcamPnP();
	
	public Mat cameraMatrix;
	public MatOfDouble distCoeffs;
	public WebcamDrawer drawer, drawer2;
	public ClassifierFormat format, smallFormat, microFormat;
	public MatOfKeyPoint keypoints, smallKeypoints, microKeypoints;
	public Mat descriptors, smallDescriptors, microDescriptors;
	public BRISK classifier;
	public FastFeatureDetector fast;
	
	public static void main(String[] args) throws IOException {
		
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		BufferedImage calibImage = ImageIO.read(new File("res/PnP/calibration.jpg"));
		Mat calibMat = bufferedImageToMat(calibImage);
		
		List<Mat> objectPointsArr = new ArrayList<>();
		List<Mat> imagePointsArr = new ArrayList<>();
		
		File imgDir = new File("res/PnP/imgs");
		File[] inImgs = imgDir.listFiles();
		instance.classifier = BRISK.create(40);
		instance.fast = FastFeatureDetector.create(25);
		for(int i = 0; i < inImgs.length; i ++) {
			File f = inImgs[i];
			//addTrain(imagePointsArr, objectPointsArr, f);
			System.out.println("Finished Training " + (i + 1) + " out of " + inImgs.length);
		}
		System.out.println();
		Mat cameraMatrix = new Mat();
		MatOfDouble distCoeffs = new MatOfDouble();
		Mat intrinsic = new Mat(3, 3, CvType.CV_32FC1);
		intrinsic.put(0, 0, 1);
		intrinsic.put(1, 1, 1);
		/**
		List<Mat> rvecs = new ArrayList<>();
		List<Mat> tvecs = new ArrayList<>();
		
		Mat objectsPoints = new Mat();
				
		Calib3d.calibrateCamera(objectPointsArr, imagePointsArr, new Size(11, 8.5), intrinsic, distCoeffs, rvecs, tvecs);
		
		System.out.println();
		
		double[] inrinsicFloat = new double[(int)intrinsic.total() + intrinsic.channels()];
		intrinsic.get(0, 0, inrinsicFloat);
		for(double f : inrinsicFloat) {
			System.out.print(f + ", ");
		}
		System.out.println();
		
		double[] distFloat = new double[(int)intrinsic.total() + intrinsic.channels()];
		distCoeffs.get(0, 0, distFloat);
		for(double f : distFloat) {
			System.out.print(f + ", ");
		}
		*/
		
		double[] inrinsicFloat = new double[] {817.063304531327,0.0,325.9485286458284,0.0,819.4690054531818,236.2597899599986,0.0,0.0,1.0,0.0};
		intrinsic.put(0, 0, inrinsicFloat);
		
		System.out.println();
		
		double[] distFloat = new double[] {-0.014680796227423968,1.3720322590501144,-0.0028429009326778093,0.0010064951672061734,-5.347658630748131};
		//distCoeffs.put(0, 0, distFloat);
		
		instance.cameraMatrix = intrinsic;
		distCoeffs.fromArray(distFloat);
		instance.distCoeffs = distCoeffs;
		
		instance.start();
		
	}
	
	public void start() throws IOException{
		WebcamDrawer drawer = new WebcamDrawer(new Size(1280, 720));
		this.drawer = drawer;
		this.drawer2 = new WebcamDrawer(new Size(1280, 720));
		Gson gson = new Gson();
		File f = new File("res/PnP/classifier.conf");
		BufferedReader reader = new BufferedReader(new FileReader(f));
		String s = reader.readLine(), fullText = "";
		while(s != null) {
			fullText += s;
			s = reader.readLine();
		}
		
		format = gson.fromJson(fullText, ClassifierFormat.class);
				
		MatOfKeyPoint keypointMat = new MatOfKeyPoint();
		keypointMat.create(instance.format.keyrows, instance.format.keycols, instance.format.keytype);
		keypointMat.put(0, 0, instance.format.keydata);
		
		Mat keypointDesc = new Mat();
		keypointDesc.create(instance.format.descrows, instance.format.desccols, instance.format.desctype);
		keypointDesc.put(0, 0, instance.format.descdata);
		
		this.keypoints = new MatOfKeyPoint(keypointMat.clone());
		this.descriptors = keypointDesc.clone();
		
		reader = new BufferedReader(new FileReader(new File("res/PnP/classifierSmall.conf")));
		s = reader.readLine();
		fullText = "";
		while(s != null) {
			fullText += s;
			s = reader.readLine();
		}
		
		smallFormat = gson.fromJson(fullText, ClassifierFormat.class);
				
		keypointMat = new MatOfKeyPoint();
		keypointMat.create(instance.smallFormat.keyrows, instance.smallFormat.keycols, instance.smallFormat.keytype);
		keypointMat.put(0, 0, instance.smallFormat.keydata);
		
		keypointDesc = new Mat();
		keypointDesc.create(instance.smallFormat.descrows, instance.smallFormat.desccols, instance.smallFormat.desctype);
		keypointDesc.put(0, 0, instance.smallFormat.descdata);
		
		this.smallKeypoints = new MatOfKeyPoint(keypointMat.clone());
		this.smallDescriptors = keypointDesc.clone();
		
		reader = new BufferedReader(new FileReader(new File("res/PnP/classifierMicro.conf")));
		s = reader.readLine();
		fullText = "";
		while(s != null) {
			fullText += s;
			s = reader.readLine();
		}
		
		microFormat = gson.fromJson(fullText, ClassifierFormat.class);
				
		keypointMat = new MatOfKeyPoint();
		keypointMat.create(instance.microFormat.keyrows, instance.microFormat.keycols, instance.microFormat.keytype);
		keypointMat.put(0, 0, instance.microFormat.keydata);
		
		keypointDesc = new Mat();
		keypointDesc.create(instance.microFormat.descrows, instance.microFormat.desccols, instance.microFormat.desctype);
		keypointDesc.put(0, 0, instance.microFormat.descdata);
		
		this.microKeypoints = new MatOfKeyPoint(keypointMat.clone());
		this.microDescriptors = keypointDesc.clone();
		
		JFrame frame = new JFrame("Webcam");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		frame.getContentPane().add(drawer, BorderLayout.CENTER);
		
		//frame.getContentPane().add(drawer2, BorderLayout.CENTER);
		
		frame.repaint();
		frame.setSize(700, 600);
		frame.setVisible(true);
		
		VideoCapture capture = new VideoCapture(700);
		System.out.println(capture.set(Videoio.CAP_PROP_EXPOSURE, -4));
		System.out.println(capture.set(Videoio.CAP_PROP_BRIGHTNESS, 125));
		//System.out.println(capture.get(Videoio.CAP_PROP_FPS));
		Mat image = new Mat();
		if(capture.isOpened()) {
			long lastTime = System.currentTimeMillis();
			while(!finished) {
				capture.read(image);
				Mat rvec = new Mat();
				Mat tvec = new Mat();
				//System.out.println("a: " + cameraMatrix.dump());
				//System.out.println();
				//System.out.println("b: " + distCoeffs.dump());
				//boolean solved = solvePnP(image, bufferedImageToMat(ImageIO.read(new File("res/PnP/calibration.jpg"))), cameraMatrix, distCoeffs, rvec, tvec);
				boolean solved = solvePnPThreaded(image, null, cameraMatrix, distCoeffs, rvec, tvec);
				if(solved) {
				
					Mat rod = new Mat();
					
					Calib3d.Rodrigues(rvec, rod);
					
					Mat transpose = new Mat();
					
					Core.transpose(rod, transpose);
					
					Mat inversed = new Mat();
					
					Core.scaleAdd(transpose, -1, Mat.zeros(transpose.size(), transpose.type()), inversed);
					
					Mat multiplyResult = new Mat();
					
					Core.gemm(inversed, tvec, 1, new Mat(), 0, multiplyResult);
					
					double[] data = new double[3];
					multiplyResult.get(0, 0, data);
					
					if(Math.abs(data[0]) > 200 || Math.abs(data[1]) > 200 || Math.abs(data[2]) > 200) {
						drawer.setMat(image);
					}else {
						double roll = Math.toDegrees(Math.atan2(-rod.get(2, 1)[0], rod.get(2, 2)[0]));
						double pitch = Math.toDegrees(Math.asin(rod.get(2, 0)[0]));
						double yaw = Math.toDegrees(Math.atan2(-rod.get(1, 0)[0], rod.get(0, 0)[0]));
						
						MatOfPoint2f imagePoints = new MatOfPoint2f();
						MatOfPoint3f objectPoints = new MatOfPoint3f();
						objectPoints.push_back(new MatOfPoint3f(new Point3(0, 0, 0)));
						objectPoints.push_back(new MatOfPoint3f(new Point3(0, 0, -1)));
						objectPoints.push_back(new MatOfPoint3f(new Point3(0, 1, 0)));
						objectPoints.push_back(new MatOfPoint3f(new Point3(1, 0, 0)));
						
						Calib3d.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);
						
						//Mat toDraw = image.clone();
						Mat toDraw = drawer.getMat();
						
						//System.out.println(imagePoints.toArray()[3].toString());
						
						Imgproc.line(toDraw, imagePoints.toArray()[3], imagePoints.toArray()[0], new Scalar(0, 0, 255), 3);
						Imgproc.line(toDraw, imagePoints.toArray()[2], imagePoints.toArray()[0], new Scalar(0, 255, 0), 3);
						Imgproc.line(toDraw, imagePoints.toArray()[1], imagePoints.toArray()[0], new Scalar(255, 0, 0), 3);
						drawer.setMat(toDraw);
						
						System.out.println("X: " + String.format("%.3f", data[0]) + " Y: " + String.format("%.3f", data[1])  + " Z: " + String.format("%.3f", data[2]) + " P: " + String.format("%.3f", pitch) + " Y: " + String.format("%.3f", yaw) + " R: " + String.format("%.3f", roll) + " | " + String.format("%.3f", (1/((System.currentTimeMillis() - lastTime)/1000.0))));
						//System.out.println("X: " + String.format("%.3f", data[0]) + " Y: " + String.format("%.3f", data[1])  + " Z: " + String.format("%.3f", data[2]) + " P: " + String.format("%.3f", pitch) + " Y: " + String.format("%.3f", yaw) + " R: " + String.format("%.3f", roll) + " | " + String.format("%.3f", (1/capTimerEnd)));
						lastTime = System.currentTimeMillis();
					}
				}else {
					//drawer.setMat(image);
				}
				
				frame.repaint();
			}
		}
	}
	
	private WebcamPnP() {
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
		
		instance.classifier.detectAndCompute(testMat, testMat, testKeypoint, testDesc);
		
		List<MatOfDMatch> matches = new ArrayList<>();
		
		BFMatcher matcher = BFMatcher.create();
		matcher.knnMatch(keypointDesc, testDesc, matches, 2);
				
		ArrayList<DMatch> arr = new ArrayList<>();
		for(MatOfDMatch mD : matches) {
			if(mD.toArray()[0].distance < (0.7 * mD.toArray()[1].distance)) {
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
	
	private static void solvePnP(File toSolve, Mat cameraMatrix, MatOfDouble distCoeffs, Mat rvec, Mat tvec) throws IOException {
		BufferedImage testImage = ImageIO.read(toSolve);
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
		
		if(arr.size() > 0) {
			
			MatOfDMatch match2 = new MatOfDMatch(arr.subList(0, (int)(arr.size() * 0.25)).toArray(new DMatch[10]));
			
			ArrayList<DMatch> computedMatches = new ArrayList<>();
			computedMatches.addAll(arr.subList(0, (int)(arr.size())));
			
			MatOfPoint2f point2fs = new MatOfPoint2f();
			MatOfPoint3f point3fs = new MatOfPoint3f();
			for(DMatch match : computedMatches) {
				KeyPoint kp = keypointMat.toArray()[match.queryIdx];
				Point3 p = new Point3((kp.pt.x / (double)format.imgWidth) * (11), (kp.pt.y / (double)format.imgHeight) * (8.5), 0);
				point3fs.push_back(new MatOfPoint3f(p));
				point2fs.push_back(new MatOfPoint2f(testKeypoint.toArray()[match.trainIdx].pt));
			}
			
			Calib3d.solvePnP(point3fs, point2fs, cameraMatrix, distCoeffs, rvec, tvec);
		
		}
	}
	
	private static boolean solvePnP(Mat toSolve, Mat calibImage, Mat cameraMatrix, MatOfDouble distCoeffs, Mat rvec, Mat tvec) throws IOException {
		Mat testMat = toSolve.clone();
		Imgproc.cvtColor(testMat, testMat, Imgproc.COLOR_BGR2GRAY);
		
		MatOfKeyPoint testKeypoint = new MatOfKeyPoint();
		Mat testDesc = new Mat();

		instance.classifier.detect(testMat, testKeypoint);
				
		instance.classifier.compute(testMat, testKeypoint, testDesc);
		
		//instance.classifier.detectAndCompute(testMat, testMat, testKeypoint, testDesc);
		
		List<MatOfDMatch> matches = new ArrayList<>();
		
		DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
		matcher.knnMatch(instance.descriptors, testDesc, matches, 2);
		
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
		instance.drawer.setMat(outImg);
		if(arr.size() > 5) {
			try {
				List<DMatch> betterMatches = new ArrayList<>();
				KeyPoint[] arr1 = instance.keypoints.toArray();
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
					KeyPoint kp = instance.keypoints.toArray()[match.queryIdx];
					Point3 p = new Point3(((kp.pt.x / (double)instance.format.imgWidth) * (11)) - (11.0/2.0), ((kp.pt.y / (double)instance.format.imgHeight) * (8.5)) - (8.5/2.0), 0);
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
	
	private static boolean solvePnPThreaded(Mat toSolve, Mat calibImage, Mat cameraMatrix, MatOfDouble distCoeffs, Mat rvec, Mat tvec) throws IOException {
		Mat testMat = toSolve.clone();
		Imgproc.cvtColor(testMat, testMat, Imgproc.COLOR_BGR2GRAY);
		
		MatOfKeyPoint testKeypoint = new MatOfKeyPoint();
		Mat testDesc = new Mat();

		instance.classifier.detect(testMat, testKeypoint);
				
		instance.classifier.compute(testMat, testKeypoint, testDesc);
		
		//instance.classifier.detectAndCompute(testMat, testMat, testKeypoint, testDesc);
		
		ThreadedMatcher fullMatcher = new ThreadedMatcher(instance.keypoints, testKeypoint, instance.descriptors, testDesc);
		ThreadedMatcher smallMatcher = new ThreadedMatcher(instance.smallKeypoints, testKeypoint, instance.smallDescriptors, testDesc);
		ThreadedMatcher microMatcher = new ThreadedMatcher(instance.microKeypoints, testKeypoint, instance.microDescriptors, testDesc);
		
		new Thread(fullMatcher).start();
		new Thread(smallMatcher).start();
		new Thread(microMatcher).start();
		
		while(!fullMatcher.isFinished() || !smallMatcher.isFinished() || !microMatcher.isFinished());
		
		ArrayList<DMatch> arr = new ArrayList<>();
		int width = 0, height = 0;
		MatOfKeyPoint keypoints;
		if(fullMatcher.getSize() > smallMatcher.getSize()) {
			if(fullMatcher.getSize() > microMatcher.getSize()) {
				arr = fullMatcher.getArr();
				width = instance.format.imgWidth;
				height = instance.format.imgHeight;
				keypoints = instance.keypoints;
				System.out.print("Full ");
			}else {
				arr = microMatcher.getArr();
				width = instance.microFormat.imgWidth;
				height = instance.microFormat.imgHeight;
				keypoints = instance.microKeypoints;
				System.out.print("Micro ");
			}
		}else {
			if(smallMatcher.getSize() > microMatcher.getSize()) {
				arr = smallMatcher.getArr();
				width = instance.smallFormat.imgWidth;
				height = instance.smallFormat.imgHeight;
				keypoints = instance.smallKeypoints;
				System.out.print("Small ");
			}else {
				arr = microMatcher.getArr();
				width = instance.microFormat.imgWidth;
				height = instance.microFormat.imgHeight;
				keypoints = instance.microKeypoints;
				System.out.print("Micro ");
			}
		}
		Mat outImg = new Mat();
		Features2d.drawKeypoints(testMat, testKeypoint, outImg);
		instance.drawer.setMat(outImg);
		if(arr.size() > 10) {
			try {
				
				ArrayList<DMatch> computedMatches = new ArrayList<>();
				if(arr.size() > 20) {
					System.out.println(arr.size());
					computedMatches.addAll(arr.subList(0, (int)(20)));
				}else {
					System.out.println("Inconsistency Detected!");
					//System.out.println(arr.size());
					computedMatches.addAll(arr);
				}
				
				MatOfPoint2f point2fs = new MatOfPoint2f();
				MatOfPoint3f point3fs = new MatOfPoint3f();
				for(DMatch match : computedMatches) {
					KeyPoint kp = keypoints.toArray()[match.queryIdx];
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
			System.out.println();
			return false;
		}
	}
}
