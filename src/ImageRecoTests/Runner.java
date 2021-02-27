package ImageRecoTests;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.*;
import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import javax.imageio.ImageIO;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

public class Runner {

	public static void main(String[] args) throws IOException {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		BufferedImage navTarget = ImageIO.read(new File("res/NavTargets/redtowergoaltarget.jpg"));
		Mat navMat = bufferedImageToMat(navTarget);
		BufferedImage fullImage = ImageIO.read(new File("res/RealTests/Test6.jpg"));
		Mat fullMat = bufferedImageToMat(fullImage);
		Imgproc.resize(navMat, navMat, new Size(navTarget.getWidth()/5, navTarget.getHeight()/5));
		Imgproc.cvtColor(navMat, navMat, Imgproc.COLOR_BGR2GRAY);
		Imgproc.cvtColor(fullMat, fullMat, Imgproc.COLOR_BGR2GRAY);
		
		System.out.println(navMat.cols() + ", " + navMat.rows());
		
		System.out.println(fullMat.cols() + ", " + fullMat.rows());
		
		MatOfKeyPoint navKeypoints = new MatOfKeyPoint();
		Mat navDescriptors = new Mat();
		
		BRISK featureDetector = BRISK.create(20);
		featureDetector.detectAndCompute(navMat, navMat, navKeypoints, navDescriptors);
		
		MatOfKeyPoint fullKeypoints = new MatOfKeyPoint();
		Mat fullDescriptors = new Mat();
		featureDetector.detectAndCompute(fullMat, fullMat, fullKeypoints, fullDescriptors);
		
		ArrayList<KeyPoint> keyPoints = new ArrayList<>();
		keyPoints.addAll(navKeypoints.toList());
		
		Mat out2 = new Mat();
		Features2d.drawKeypoints(fullMat, fullKeypoints, out2);
		Image i2 = HighGui.toBufferedImage(out2);
		File f2 = new File("res/ImageReco/outKeypoints.png");
		f2.createNewFile();
		ImageIO.write(toBufferedImage(i2), "PNG", f2);
		
		Mat out3 = new Mat();
		Features2d.drawKeypoints(navMat, navKeypoints, out3);
		Image i3 = HighGui.toBufferedImage(out3);
		File f3 = new File("res/ImageReco/outNavKeypoints.png");
		f2.createNewFile();
		ImageIO.write(toBufferedImage(i3), "PNG", f3);
		
		System.out.println("Found Keypoints");
		
		//MatOfDMatch matches = new MatOfDMatch();
		List<MatOfDMatch> matches = new ArrayList<>();
		
		BFMatcher matcher = BFMatcher.create();
		//matcher.match(navDescriptors, fullDescriptors, matches);
		matcher.knnMatch(navDescriptors, fullDescriptors, matches, 2);
				
		ArrayList<DMatch> arr = new ArrayList<>();
		for(MatOfDMatch mD : matches) {
			if(mD.toArray()[0].distance < (0.8 * mD.toArray()[1].distance)) {
				arr.add(mD.toArray()[0]);
			}
			//arr.add(mD.toArray()[0]);
			//arr.add(mD.toArray()[1]);
		}
		
		List<DMatch> betterMatches = new ArrayList<>();
		KeyPoint[] arr1 = navKeypoints.toArray();
		KeyPoint[] arr2 = fullKeypoints.toArray();
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
		System.out.println(betterMatches.size());
		System.out.println(arr.size());
		if(betterMatches.size() > 5) {
			arr.clear();
			arr.addAll(betterMatches);
		}
		
		Collections.sort(arr, new Comparator<DMatch>() {

			@Override
			public int compare(DMatch o1, DMatch o2) {
				return (int)(o1.distance - o2.distance);
			}
			
		});
		MatOfDMatch match2 = new MatOfDMatch(arr.subList(0, (int)(arr.size())).toArray(new DMatch[10]));
		
		//System.out.println(navKeypoints.cols() + ", " + navKeypoints.rows() + " | " + fullKeypoints.cols() + ", " + fullKeypoints.rows());
		//System.out.println(navKeypoints.dump());
		
		Mat out = new Mat();
		Features2d.drawMatches(navMat, navKeypoints, fullMat, fullKeypoints, match2, out);
		Image i = HighGui.toBufferedImage(out);
		File f = new File("res/ImageReco/outMatches.png");
		f.createNewFile();
		ImageIO.write(toBufferedImage(i), "PNG", f);
		
		BufferedImage buffOut = fullImage;
		Graphics g = buffOut.getGraphics();
		g.setColor(Color.black);
		for(DMatch match : arr.subList(0, arr.size())) {
			KeyPoint kp = fullKeypoints.toArray()[match.trainIdx];
			g.fillRect((int)kp.pt.x, (int)kp.pt.y, 10, 10);
		}
		
		ImageIO.write(buffOut, "PNG", new File("res/ImageReco/outTestImg.png"));
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

}
