/**
 *    KNN.java
 *    Copyright 2004 Liangxiao Jiang
 **/

package weka.classifiers.caitao;

import weka.classifiers.*;
import weka.core.*;

/**
 * Implement an KNN classifier.
 */
public class KNN extends Classifier
{

	private static final long serialVersionUID = -4146766600694352056L;

	/** The training instances used for classification. */
	private Instances m_Train;

	/** The number of neighbors to use for classification. */
	private int m_kNN;

	/**
	 * Builds KNN classifier.
	 *
	 * @param data the training data
	 * @exception Exception if classifier can't be built successfully
	 */
	public void buildClassifier(Instances data) throws Exception
	{

		//initial data
		m_Train = new Instances(data);  // lazy 体现在 buildClassifier 阶段什么都不做
		m_kNN = 3;
	}

	/**
	 * Computes class distribution for a test instance.
	 *
	 * @param instance the instance for which distribution is to be computed
	 * @return the class distribution for the given instance
	 */
	public double[] distributionForInstance(Instance instance) throws Exception
	{

		NeighborList neighborlist = findNeighbors(instance, m_kNN);
		return computeDistribution(neighborInstances(neighborlist), instance);
	}

	/**
	 * Build the list of nearest k neighbors to the given test instance.
	 *
	 * @param instance the instance to search for neighbors
	 * @return a list of neighbors
	 */
	private NeighborList findNeighbors(Instance instance, int kNN)
	{

		double distance;
		NeighborList neighborlist = new NeighborList(kNN);
		for (int i = 0; i < m_Train.numInstances(); i++)
		{
			Instance trainInstance = m_Train.instance(i);
			distance = distance(instance, trainInstance);
			if (neighborlist.isEmpty() || i < kNN || distance <= neighborlist.m_Last.m_Distance)
			{
				neighborlist.insertSorted(distance, trainInstance);
			}
		}
		return neighborlist;

	}

	/**
	 * Turn the list of nearest neighbors into a probability distribution
	 *
	 * @param neighborlist the list of nearest neighboring instances
	 * @return the probability distribution                                              // TODO 笔误
	 */
	private Instances neighborInstances(NeighborList neighborlist) throws Exception
	{

		Instances neighborInsts = new Instances(m_Train, neighborlist.currentLength());  // TODO Instances java doc中的header info是指什么？
		if (!neighborlist.isEmpty())
		{
			NeighborNode current = neighborlist.m_First;
			while (current != null)
			{
				neighborInsts.add(current.m_Instance);
				current = current.m_Next;
			}
		}
		return neighborInsts;

	}

	/**
	 * Calculates the distance between two instances
	 *
	 * @param first the first instance
	 * @param second the second instance
	 * @return the distance between the two given instances
	 */
	private double distance(Instance first, Instance second)
	{

		double distance = 0;
		for (int i = 0; i < m_Train.numAttributes(); i++)
		{
			if (i == m_Train.classIndex())
				continue;
			if ((int) first.value(i) != (int) second.value(i))
			{
				distance += 1;                                  // 适用于nominal，不同加一，相同不加一
			}
		}
		return distance;
	}

	/**
	 * Compute the distribution.
	 *
	 * @param data the training data                                       // TODO 参数有误
	 * @exception Exception if classifier can't be built successfully
	 */
	private double[] computeDistribution(Instances data, Instance instance) throws Exception
	{

		int numClasses = data.numClasses();
		double[] probs = new double[numClasses];
		double[] classCounts = new double[numClasses];
		int numInstances = data.numInstances();
		for (int i = 0; i < numInstances; i++)
		{
			int classVal = (int) data.instance(i).classValue();
			classCounts[classVal]++;
		}
		for (int i = 0; i < numClasses; i++)
		{
			probs[i] = (classCounts[i] + 1.0) / (numInstances + numClasses);
		}
		Utils.normalize(probs);
		return probs;
	}

	/**
	 * Main method.
	 *
	 * @param args the options for the classifier                         // TODO 如何设置options?
	 */
	public static void main(String[] args)
	{

		try
		{
			System.out.println(Evaluation.evaluateModel(new KNN(), args));
		}
		catch (Exception e)
		{
			System.err.println(e.getMessage());
		}
	}

	/*
	 * A class for storing data about a neighboring instance
	 */
	private class NeighborNode
	{

		/** The neighbor instance */
		private Instance m_Instance;

		/** The distance from the current instance to this neighbor */
		private double m_Distance;

		/** A link to the next neighbor instance */
		private NeighborNode m_Next;

		/**
		 * Create a new neighbor node.
		 *
		 * @param distance the distance to the neighbor
		 * @param instance the neighbor instance
		 * @param next the next neighbor node
		 */
		public NeighborNode(double distance, Instance instance, NeighborNode next)
		{
			m_Distance = distance;
			m_Instance = instance;
			m_Next = next;
		}

		/**
		 * Create a new neighbor node that doesn't link to any other nodes.
		 *
		 * @param distance the distance to the neighbor
		 * @param instance the neighbor instance
		 */
		public NeighborNode(double distance, Instance instance)
		{

			this(distance, instance, null);
		}
	}

	/*
	 * A class for a linked list to store the nearest k neighbors to an instance.
	 */
	private class NeighborList
	{

		/** The first node in the list */
		private NeighborNode m_First;

		/** The last node in the list */
		private NeighborNode m_Last;

		/** The number of nodes to attempt to maintain in the list */
		private int m_Length = 1;  // KNN 算法中的 K

		/**
		 * Creates the neighborlist with a desired length
		 *
		 * @param length the length of list to attempt to maintain
		 */
		public NeighborList(int length)
		{

			m_Length = length;
		}

		/**
		 * Gets whether the list is empty.
		 *
		 * @return true if so
		 */
		public boolean isEmpty()
		{

			return (m_First == null);
		}

		/**
		 * Gets the current length of the list.
		 *
		 * @return the current length of the list
		 */
		public int currentLength()
		{

			int i = 0;
			NeighborNode current = m_First;
			while (current != null)
			{
				i++;
				current = current.m_Next;
			}
			return i;
		}

		/**
		 * Inserts an instance neighbor into the list, maintaining the list sorted by distance.
		 *
		 * @param distance the distance to the instance
		 * @param instance the neighboring instance
		 */
		public void insertSorted(double distance, Instance instance)
		{

			if (isEmpty())
			{
				m_First = m_Last = new NeighborNode(distance, instance);
			}
			else
			{
				NeighborNode current = m_First;
				if (distance < m_First.m_Distance)
				{// Insert at head
					m_First = new NeighborNode(distance, instance, m_First);
				}
				else
				{ // Insert further down the list
					for (; (current.m_Next != null) && (current.m_Next.m_Distance < distance); current = current.m_Next)
						;
					current.m_Next = new NeighborNode(distance, instance, current.m_Next);
					if (current.equals(m_Last))
					{
						m_Last = current.m_Next;
					}
				}

				// Trip down the list until we've got k list elements (or more if the distance to the last elements is the same).
				int valcount = 0;
				for (current = m_First; current.m_Next != null; current = current.m_Next)
				{
					valcount++;
					if ((valcount >= m_Length) && (current.m_Distance != current.m_Next.m_Distance))
					{
						m_Last = current;
						current.m_Next = null;
						break;
					}
				}
			}
		}

		/**
		 * Prunes the list to contain the k nearest neighbors. If there are multiple neighbors at the k'th distance, all will be kept.
		 *
		 * @param k the number of neighbors to keep in the list.
		 */
		@SuppressWarnings("unused")
		public void pruneToK(int k)
		{

			if (isEmpty())
			{
				return;
			}
			if (k < 1)
			{
				k = 1;
			}
			int currentK = 0;
			double currentDist = m_First.m_Distance;
			NeighborNode current = m_First;
			for (; current.m_Next != null; current = current.m_Next)
			{
				currentK++;
				currentDist = current.m_Distance;
				if ((currentK >= k) && (currentDist != current.m_Next.m_Distance))
				{
					m_Last = current;
					current.m_Next = null;
					break;
				}
			}
		}

	}

}
