/***************************************************************************************************
 * One way to check whether the recommender returns good results is by doing a hold-out test.
 * We partition our data set into two sets: a training set consisting of 90% of the data and 
 * a test set consisting of 10%. Then we train our recommender using the training set 
 * and look how well it predicts the unknown interactions in the test set.
 * Note: if you run this test multiple times, you will get different results, 
 * because the splitting into training set and test set is done randomly.
 * This evaluation needs a big data set to give good results
 * **************************************************************************************************/

package com.predictionmarketing.RecommenderApp;

import java.io.File;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

public class EvaluateRecommender {

	public static void main(String[] args) throws Exception{
		DataModel model = new FileDataModel(new File("data/dataset.csv"));
		RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
		RecommenderBuilder builder = new MyRecommenderBuilder();
		double result = evaluator.evaluate(builder, null, model, 0.9, 1.0);
		System.out.println(result);

	}

}

class MyRecommenderBuilder implements RecommenderBuilder{
	
	public Recommender buildRecommender(DataModel dataModel) throws TasteException{
	UserSimilarity similarity = new PearsonCorrelationSimilarity(dataModel);
	UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.1, similarity, dataModel);
	return new GenericUserBasedRecommender(dataModel, neighborhood, similarity);
	}
}
