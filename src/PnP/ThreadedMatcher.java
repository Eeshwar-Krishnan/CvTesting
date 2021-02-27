package PnP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.features2d.DescriptorMatcher;

public class ThreadedMatcher implements Runnable{
	Mat queryDesc, trainDesc;
	MatOfKeyPoint queryKey, trainKey;
	AtomicBoolean finished;
	AtomicInteger size;
	ArrayList<DMatch> arr = new ArrayList<>();
	
	public ThreadedMatcher(MatOfKeyPoint queryKey, MatOfKeyPoint trainKey, Mat queryDesc, Mat trainDesc) {
		this.queryDesc = queryDesc;
		this.trainDesc = trainDesc;
		this.queryKey = queryKey;
		this.trainKey = trainKey;
		finished = new AtomicBoolean(false);
		size = new AtomicInteger(0);
	}

	@Override
	public void run() {
		List<MatOfDMatch> matches = new ArrayList<>();
		
		DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
		matcher.knnMatch(queryDesc, trainDesc, matches, 2);
		
		arr = new ArrayList<>();
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
		if(arr.size() > 5) {
			List<DMatch> betterMatches = new ArrayList<>();
			KeyPoint[] arr1 = queryKey.toArray();
			KeyPoint[] arr2 = trainKey.toArray();
			MatOfPoint2f pts1Mat = new MatOfPoint2f();
			MatOfPoint2f pts2Mat = new MatOfPoint2f();
			for(int i = 0; i < arr.size(); i ++) {
				pts1Mat.push_back(new MatOfPoint2f(arr1[arr.get(i).queryIdx].pt));
				pts2Mat.push_back(new MatOfPoint2f(arr2[arr.get(i).trainIdx].pt));
			}
			Mat mask = new Mat();
			Calib3d.findHomography(pts1Mat, pts2Mat, Calib3d.RANSAC, 10, mask);
			
			for(int i = (arr.size()-1); i >= 0; i --) {
				if(mask.get(i, 0)[0] != 0.0) {
					betterMatches.add(arr.get(i));
				}
			}
			if(betterMatches.size() > 5) {
				arr.clear();
				arr.addAll(betterMatches);
			}
		}
		size.set(arr.size());
		finished.set(true);
	}
	
	
	
	public boolean isFinished() {
		return finished.get();
	}
	
	public int getSize() {
		return size.get();
	}
	
	public ArrayList<DMatch> getArr(){
		return arr;
	}

}
