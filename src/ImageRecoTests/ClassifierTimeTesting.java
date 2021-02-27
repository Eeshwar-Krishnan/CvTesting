package ImageRecoTests;

import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import javax.imageio.ImageIO;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.BRISK;
import org.opencv.features2d.FastFeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.ORB;
import org.opencv.features2d.SIFT;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

import com.google.gson.Gson;

import PnP.ClassifierFormat;

public class ClassifierTimeTesting {

	public static void main(String[] args) throws IOException {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		BufferedImage navTarget = ImageIO.read(new File("res/navTarget.jpg"));
		Mat navMat = bufferedImageToMat(navTarget);
		//Imgproc.cvtColor(navMat, navMat, Imgproc.COLOR_BGR2GRAY);
				
		MatOfKeyPoint navKeypoints = new MatOfKeyPoint();
		Mat navDescriptors = new Mat();
		
		BRISK featureDetector = BRISK.create(50);
		long allBriskStart = System.nanoTime();
		featureDetector.detectAndCompute(navMat, navMat, navKeypoints, navDescriptors);
		long allBriskEnd = System.nanoTime();
		System.out.println(navKeypoints.total() + " | " + navDescriptors.total());
		navKeypoints = new MatOfKeyPoint();
		navDescriptors = new Mat();
				
		FastFeatureDetector fast = FastFeatureDetector.create(10);
		
		long hybridStart = System.nanoTime();
		fast.detect(navMat, navKeypoints);
		featureDetector.compute(navMat, navKeypoints, navDescriptors);
		long hybridEnd = System.nanoTime();
		System.out.println("BRISK: " + (allBriskEnd - allBriskStart)/(1.0e+9) + " hybrid: " + (hybridEnd - hybridStart)/(1.0e+9));
		System.out.println(navKeypoints.total() + " | " + navDescriptors.total());
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
