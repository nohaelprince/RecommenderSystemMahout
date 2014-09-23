/* ****************************************************************************************************
 * objective: This program creates a use-based recommender system using Apache Mahout java-based library.
 * The idea behind this approach is that when we want to compute recommendations for a particular users,
 * we look for other users with a similar taste and pick the recommendations from their items. 
 * Created by: Noha A. Elprince
 * Reference: http://mahout.apache.org/users/recommender/userbased-5-minutes.html
 * ****************************************************************************************************/
package com.predictionmarketing.RecommenderApp;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;


public class App 
{
    public static void main( String[] args ) throws Exception
    {
    	// Load data from the file
    	DataModel model = new FileDataModel(new File("data/dataset.csv"));
    	// For finding similar users, Compute the correlation coefficient between their interactions
    	UserSimilarity similarity = new PearsonCorrelationSimilarity(model);
    	// Define which similar users we want to leverage for the recommender.
    	// we'll use all that have a similarity greater than 0.1. 
    	// This is implemented via a ThresholdUserNeighborhood:
    	UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.1, similarity, model);
    	// instantiate a recommender
    	UserBasedRecommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);
    	// ask the recommender for recommendations now
    	// we want to get 3 items recommended for the user with userID 2
    	List<RecommendedItem> recommendations = recommender.recommend(2, 3);
    	for (RecommendedItem recommendation : recommendations) {
    	  System.out.println(recommendation);
    	}
    }
}
