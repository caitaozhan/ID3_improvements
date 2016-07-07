/*
 *    NB.java
 *    Copyright 2005 Liangxiao Jiang
 */

package weka.classifiers.caitao;

import weka.core.*;
import weka.classifiers.*;

/**
 * Implement the NB classifier.
 */
public class NB extends Classifier
{

	private static final long serialVersionUID = 4283978271381326861L;

	/** The number of class and each attribute value occurs in the dataset */
	private double[][] m_ClassAttCounts;

	/** The number of each class value occurs in the dataset */
	private double[] m_ClassCounts;

	/** The number of values for each attribute in the dataset */
	private int[] m_NumAttValues;

	/** The starting index of each attribute in the dataset */
	private int[] m_StartAttIndex;

	/** The number of values for all attributes in the dataset */
	private int m_TotalAttValues;

	/** The number of classes in the dataset */
	private int m_NumClasses;

	/** The number of attributes including class in the dataset */
	private int m_NumAttributes;

	/** The number of instances in the dataset */
	private int m_NumInstances;

	/** The index of the class attribute in the dataset */
	private int m_ClassIndex;

	/**
	 * Generates the classifier.
	 *
	 * @param instances set of instances serving as training data
	 * @exception Exception if the classifier has not been generated successfully
	 */
	public void buildClassifier(Instances instances) throws Exception
	{

		// reset variable
		m_NumClasses = instances.numClasses();
		m_ClassIndex = instances.classIndex();
		m_NumAttributes = instances.numAttributes();
		m_NumInstances = instances.numInstances();
		m_TotalAttValues = 0;
		// allocate space for attribute reference arrays
		m_StartAttIndex = new int[m_NumAttributes];
		m_NumAttValues = new int[m_NumAttributes];
		// set the starting index of each attribute and the number of values for
		// each attribute and the total number of values for all attributes(not including class).
		for (int i = 0; i < m_NumAttributes; i++)
		{
			if (i != m_ClassIndex)
			{
				m_StartAttIndex[i] = m_TotalAttValues;
				m_NumAttValues[i] = instances.attribute(i).numValues();
				m_TotalAttValues += m_NumAttValues[i];
			}
			else
			{
				m_StartAttIndex[i] = -1;
				m_NumAttValues[i] = m_NumClasses;
			}
		}
		// allocate space for counts and frequencies
		m_ClassCounts = new double[m_NumClasses];
		m_ClassAttCounts = new double[m_NumClasses][m_TotalAttValues];
		// Calculate the counts
		for (int k = 0; k < m_NumInstances; k++)
		{
			int classVal = (int) instances.instance(k).classValue();
			m_ClassCounts[classVal]++;
			int[] attIndex = new int[m_NumAttributes];
			for (int i = 0; i < m_NumAttributes; i++)
			{
				if (i == m_ClassIndex)
				{
					attIndex[i] = -1;
				}
				else
				{
					attIndex[i] = m_StartAttIndex[i] + (int) instances.instance(k).value(i);
					m_ClassAttCounts[classVal][attIndex[i]]++;
				}
			}
		}
	}

	/**
	 * Calculates the class membership probabilities for the given test instance
	 *
	 * @param instance the instance to be classified
	 * @return predicted class probability distribution
	 * @exception Exception if there is a problem generating the prediction
	 */
	public double[] distributionForInstance(Instance instance) throws Exception
	{

		//Definition of local variables
		double[] probs = new double[m_NumClasses];
		// store instance's att values in an int array
		int[] attIndex = new int[m_NumAttributes];
		for (int att = 0; att < m_NumAttributes; att++)
		{
			if (att == m_ClassIndex)
				attIndex[att] = -1;
			else
				attIndex[att] = m_StartAttIndex[att] + (int) instance.value(att);
		}
		// calculate probabilities for each possible class value
		for (int classVal = 0; classVal < m_NumClasses; classVal++)
		{
			probs[classVal] = (m_ClassCounts[classVal] + 1.0) / (m_NumInstances + m_NumClasses);
			for (int att = 0; att < m_NumAttributes; att++)
			{
				if (attIndex[att] == -1)
					continue;
				probs[classVal] *= (m_ClassAttCounts[classVal][attIndex[att]] + 1.0) / (m_ClassCounts[classVal] + m_NumAttValues[att]);
			}
		}

		Utils.normalize(probs);
		return probs;
	}

	/**
	 * Main method for testing this class.
	 *
	 * @param argv the options
	 */
	public static void main(String[] argv)
	{
		try
		{
			System.out.println(Evaluation.evaluateModel(new NB(), argv));
		}
		catch (Exception e)
		{
			e.printStackTrace();
			System.err.println(e.getMessage());
		}
	}

}
