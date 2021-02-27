package Robot;

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
import org.opencv.core.Size;
import org.opencv.features2d.BRISK;
import org.opencv.features2d.FastFeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

import com.google.gson.Gson;

import PnP.ClassifierFormat;
import Testing.FormatTesting;

public class ImageClassifier {

	public static void main(String[] args) throws IOException {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		BufferedImage full = ImageIO.read(new File("res/NavTargets/redtowergoaltarget.jpg"));
		serialize(full);
	}
	
	public static void serialize(BufferedImage full) throws IOException {
		RawClassifierFormat format = new RawClassifierFormat();
		Mat fullMat = bufferedImageToMat(full);
		Mat micro = new Mat();
		Mat mini = new Mat();
		Imgproc.resize(fullMat, mini, new Size(fullMat.size().width/2, fullMat.size().height/2));
		Imgproc.resize(fullMat, micro, new Size(fullMat.size().width/4, fullMat.size().height/4));
		format.mini = serializeOne(mini, mini.size());
		format.full = serializeOne(fullMat, fullMat.size());
		format.micro = serializeOne(micro, micro.size());
		
		File outFile = new File("res/Classifiers/classifier.conf");
		outFile.createNewFile();
		BufferedWriter writer = new BufferedWriter(new FileWriter(outFile));
		Gson gson = new Gson();
		writer.write(gson.toJson(format));
		
		writer.flush();
		writer.close();
		
		System.out.println("Wrote Config File!");
	}
	
	public static RawKeypointFormat serializeOne(Mat image, Size size) throws IOException {
		Mat navMat = image;
		Imgproc.cvtColor(navMat, navMat, Imgproc.COLOR_BGR2GRAY);
		
		MatOfKeyPoint navKeypoints = new MatOfKeyPoint();
		Mat navDescriptors = new Mat();
		
		BRISK featureDetector = BRISK.create(40);
		FastFeatureDetector fast = FastFeatureDetector.create(25);
		featureDetector.detect(navMat, navKeypoints);
		featureDetector.compute(navMat, navKeypoints, navDescriptors);
		
		System.out.println("Found Keypoints");
		
		Mat out = new Mat();
		Features2d.drawKeypoints(navMat, navKeypoints, out);
		Image i = HighGui.toBufferedImage(out);
		File f = new File("res/Classifiers/outNavKeypoints.png");
		ImageIO.write(toBufferedImage(i), "PNG", f);
		
		float[] data = new float[(int)navKeypoints.total() * navKeypoints.channels()];
		navKeypoints.get(0, 0, data);
		RawKeypointFormat format = new RawKeypointFormat();
		format.keydata = data;
		format.keycols = navKeypoints.cols();
		format.keyrows = navKeypoints.rows();
		format.keytype = navKeypoints.type();
		
		byte[] data2 = new byte[(int)navDescriptors.total() * navDescriptors.channels()];
		navDescriptors.get(0, 0, data2);
		format.descdata = data2;
		format.desccols = navDescriptors.cols();
		format.descrows = navDescriptors.rows();
		format.desctype = navDescriptors.type();
		
		format.imgWidth = (int) size.width;
		format.imgHeight = (int) size.height;
		
		return format;
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
