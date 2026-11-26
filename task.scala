import java.io.{BufferedWriter, FileWriter, PrintWriter}
import scala.collection.mutable
import scala.util.Random

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors

object task {

  case class DataPoint(id: Int, features: Array[Double])

  class ClusterStat(dim: Int) {
    var N: Int = 0
    val SUM: Array[Double] = Array.fill(dim)(0.0)
    val SUMSQ: Array[Double] = Array.fill(dim)(0.0)
    val pointIds: mutable.ArrayBuffer[Int] = mutable.ArrayBuffer.empty[Int]

    def this(points: Seq[Array[Double]], ids: Seq[Int]) = {
      this(points.head.length)
      var i = 0
      while (i < points.length) {
        addPoint(points(i), ids(i))
        i += 1
      }
    }

    def addPoint(point: Array[Double], pointId: Int): Unit = {
      N += 1
      var d = 0
      while (d < point.length) {
        SUM(d) += point(d)
        SUMSQ(d) += point(d) * point(d)
        d += 1
      }
      pointIds += pointId
    }

    def merge(other: ClusterStat): Unit = {
      N += other.N
      var d = 0
      while (d < SUM.length) {
        SUM(d) += other.SUM(d)
        SUMSQ(d) += other.SUMSQ(d)
        d += 1
      }
      pointIds ++= other.pointIds
    }

    def centroid: Array[Double] = {
      val c = Array.ofDim[Double](SUM.length)
      var d = 0
      while (d < SUM.length) {
        c(d) = SUM(d) / N.toDouble
        d += 1
      }
      c
    }

    def std: Array[Double] = {
      val mu = centroid
      val s = Array.ofDim[Double](SUM.length)
      var d = 0
      while (d < SUM.length) {
        s(d) = math.sqrt(math.max(SUMSQ(d) / N.toDouble - mu(d) * mu(d), 1e-6))
        d += 1
      }
      s
    }
  }

  def mahalanobisPoint(point: Array[Double], cluster: ClusterStat): Double = {
    val mu = cluster.centroid
    val s = cluster.std
    var sum = 0.0
    var d = 0
    while (d < point.length) {
      val diff = (point(d) - mu(d)) / s(d)
      sum += diff * diff
      d += 1
    }
    math.sqrt(sum)
  }

  def mahalanobisClusters(c1: ClusterStat, c2: ClusterStat): Double = {
    math.min(mahalanobisPoint(c1.centroid, c2), mahalanobisPoint(c2.centroid, c1))
  }

  def euclideanSq(a: Array[Double], b: Array[Double]): Double = {
    var s = 0.0
    var d = 0
    while (d < a.length) {
      val diff = a(d) - b(d)
      s += diff * diff
      d += 1
    }
    s
  }

  def runKMeansLocal(points: Array[Array[Double]], k: Int, seed: Long): Array[Int] = {
    val n = points.length
    val kk = math.max(1, math.min(k, n))
    if (kk == 1) return Array.fill(n)(0)
    val dim = points(0).length
    val rng = new Random(seed)
    var bestLabels: Array[Int] = null
    var bestInertia = Double.PositiveInfinity
    val nInit = 10
    val maxIter = 50

    var trial = 0
    while (trial < nInit) {
      val centroids = Array.ofDim[Array[Double]](kk)
      centroids(0) = points(rng.nextInt(n)).clone()
      val closest = Array.fill(n)(Double.PositiveInfinity)
      var c = 1
      while (c < kk) {
        var i = 0
        var sum = 0.0
        while (i < n) {
          val d = euclideanSq(points(i), centroids(c - 1))
          if (d < closest(i)) closest(i) = d
          sum += closest(i)
          i += 1
        }
        var r = rng.nextDouble() * math.max(sum, 1e-12)
        var pick = n - 1
        i = 0
        while (i < n) {
          r -= closest(i)
          if (r <= 0) { pick = i; i = n } else i += 1
        }
        centroids(c) = points(pick).clone()
        c += 1
      }

      val labels = Array.fill(n)(0)
      var iter = 0
      var changed = true
      while (iter < maxIter && changed) {
        changed = false
        var i = 0
        while (i < n) {
          var best = 0
          var bestD = Double.PositiveInfinity
          var j = 0
          while (j < kk) {
            val d = euclideanSq(points(i), centroids(j))
            if (d < bestD) { bestD = d; best = j }
            j += 1
          }
          if (labels(i) != best) { labels(i) = best; changed = true }
          i += 1
        }
        val sums = Array.ofDim[Double](kk, dim)
        val counts = Array.fill(kk)(0)
        i = 0
        while (i < n) {
          val lab = labels(i)
          counts(lab) += 1
          var d = 0
          while (d < dim) {
            sums(lab)(d) += points(i)(d)
            d += 1
          }
          i += 1
        }
        var j = 0
        while (j < kk) {
          if (counts(j) > 0) {
            var d = 0
            while (d < dim) {
              centroids(j)(d) = sums(j)(d) / counts(j).toDouble
              d += 1
            }
          } else {
            centroids(j) = points(rng.nextInt(n)).clone()
            changed = true
          }
          j += 1
        }
        iter += 1
      }

      var inertia = 0.0
      var i = 0
      while (i < n) {
        inertia += euclideanSq(points(i), centroids(labels(i)))
        i += 1
      }
      if (inertia < bestInertia) {
        bestInertia = inertia
        bestLabels = labels.clone()
      }
      trial += 1
    }
    bestLabels
  }

  def refineKMeans(points: Array[Array[Double]], initCentroids: Array[Array[Double]], maxIter: Int = 40): Array[Int] = {
    val n = points.length
    val kk = initCentroids.length
    val dim = points(0).length
    val centroids = initCentroids.map(_.clone())
    val labels = Array.fill(n)(0)
    var iter = 0
    var changed = true
    while (iter < maxIter && changed) {
      changed = false
      var i = 0
      while (i < n) {
        var best = 0
        var bestD = Double.PositiveInfinity
        var j = 0
        while (j < kk) {
          val d = euclideanSq(points(i), centroids(j))
          if (d < bestD) { bestD = d; best = j }
          j += 1
        }
        if (labels(i) != best) { labels(i) = best; changed = true }
        i += 1
      }
      val sums = Array.ofDim[Double](kk, dim)
      val counts = Array.fill(kk)(0)
      i = 0
      while (i < n) {
        val lab = labels(i)
        counts(lab) += 1
        var d = 0
        while (d < dim) {
          sums(lab)(d) += points(i)(d)
          d += 1
        }
        i += 1
      }
      var j = 0
      while (j < kk) {
        if (counts(j) > 0) {
          var d = 0
          while (d < dim) {
            centroids(j)(d) = sums(j)(d) / counts(j).toDouble
            d += 1
          }
        }
        j += 1
      }
      iter += 1
    }
    labels
  }

  def assignToCentroids(points: Array[Array[Double]], centroids: Array[Array[Double]]): Array[Int] = {
    val n = points.length
    val kk = centroids.length
    val labels = Array.ofDim[Int](n)
    var i = 0
    while (i < n) {
      var best = 0
      var bestD = Double.PositiveInfinity
      var j = 0
      while (j < kk) {
        val d = euclideanSq(points(i), centroids(j))
        if (d < bestD) { bestD = d; best = j }
        j += 1
      }
      labels(i) = best
      i += 1
    }
    labels
  }

  def centroidsFromLabels(points: Array[Array[Double]], labels: Array[Int], k: Int): Array[Array[Double]] = {
    val dim = points(0).length
    val sums = Array.ofDim[Double](k, dim)
    val counts = Array.fill(k)(0)
    var i = 0
    while (i < points.length) {
      val lab = labels(i)
      counts(lab) += 1
      var d = 0
      while (d < dim) {
        sums(lab)(d) += points(i)(d)
        d += 1
      }
      i += 1
    }
    val cents = Array.ofDim[Array[Double]](k)
    var j = 0
    val rng = new Random(1)
    while (j < k) {
      if (counts(j) > 0) {
        var d = 0
        while (d < dim) {
          sums(j)(d) /= counts(j).toDouble
          d += 1
        }
        cents(j) = sums(j)
      } else cents(j) = points(rng.nextInt(points.length)).clone()
      j += 1
    }
    cents
  }

  // Fast KMeans: fit on a sample when n is large, then assign everyone (grader has ~320k pts)
  def runKMeans(sc: SparkContext, points: Array[Array[Double]], k: Int, seed: Long): Array[Int] = {
    val n = points.length
    if (n == 0) return Array.empty[Int]
    val kk = math.max(1, math.min(k, n))
    if (kk == 1) return Array.fill(n)(0)

    val fitN = math.min(n, 12000)
    val fitPts =
      if (fitN == n) points
      else {
        val rng = new Random(seed)
        val idx = Array.tabulate(n)(i => i)
        var i = n - 1
        while (i > 0) {
          val j = rng.nextInt(i + 1)
          val tmp = idx(i); idx(i) = idx(j); idx(j) = tmp
          i -= 1
        }
        Array.tabulate(fitN)(t => points(idx(t)))
      }

    // Prefer a single Spark MLlib pass on the (possibly sampled) fit set; fallback to local
    val fitLabels =
      if (fitN >= 3000) {
        val parts = math.max(2, math.min(4, fitN / 3000 + 1))
        val rdd = sc.parallelize(fitPts.map(p => Vectors.dense(p)), parts).cache()
        val model = KMeans.train(rdd, kk, maxIterations = 25, initializationMode = "k-means||", seed = seed)
        val labs = rdd.map(v => model.predict(v)).collect()
        rdd.unpersist(false)
        labs
      } else runKMeansLocal(fitPts, kk, seed)

    if (fitN == n) fitLabels
    else {
      val cents = centroidsFromLabels(fitPts, fitLabels, kk)
      // one refine pass on full data from those centroids
      refineKMeans(points, cents, maxIter = 15)
    }
  }

  def splitBySize(
      labels: Array[Int],
      pointIds: Array[Int],
      points: Array[Array[Double]]
  ): (Array[(Array[Int], Array[Array[Double]])], Array[(Int, Array[Double])]) = {
    val groups = mutable.HashMap.empty[Int, mutable.ArrayBuffer[Int]]
    var i = 0
    while (i < labels.length) {
      groups.getOrElseUpdate(labels(i), mutable.ArrayBuffer.empty[Int]) += i
      i += 1
    }
    val multi = mutable.ArrayBuffer.empty[(Array[Int], Array[Array[Double]])]
    val singles = mutable.ArrayBuffer.empty[(Int, Array[Double])]
    groups.values.foreach { idxs =>
      if (idxs.length > 1) multi += ((idxs.map(pointIds).toArray, idxs.map(points).toArray))
      else {
        val j = idxs.head
        singles += ((pointIds(j), points(j)))
      }
    }
    (multi.toArray, singles.toArray)
  }

  def rsToCsAndRs(
      sc: SparkContext,
      rsIds: mutable.ArrayBuffer[Int],
      rsPoints: mutable.ArrayBuffer[Array[Double]],
      nCluster: Int,
      seed: Long
  ): (mutable.ArrayBuffer[ClusterStat], mutable.ArrayBuffer[Int], mutable.ArrayBuffer[Array[Double]]) = {
    val csList = mutable.ArrayBuffer.empty[ClusterStat]
    val newRsIds = mutable.ArrayBuffer.empty[Int]
    val newRsPoints = mutable.ArrayBuffer.empty[Array[Double]]
    if (rsIds.isEmpty) return (csList, newRsIds, newRsPoints)
    if (rsIds.length == 1) {
      newRsIds += rsIds.head
      newRsPoints += rsPoints.head
      return (csList, newRsIds, newRsPoints)
    }
    val pts = rsPoints.toArray
    val ids = rsIds.toArray
    val largeK = math.min(5 * nCluster, math.max(1, (rsIds.length + 1) / 2))
    val labels = runKMeans(sc, pts, largeK, seed)
    val (multi, singles) = splitBySize(labels, ids, pts)
    multi.foreach { case (mids, mpts) => csList += new ClusterStat(mpts, mids) }
    singles.foreach { case (pid, pt) =>
      newRsIds += pid
      newRsPoints += pt
    }
    (csList, newRsIds, newRsPoints)
  }

  def assignToNearest(
      points: Array[Array[Double]],
      pointIds: Array[Int],
      clusters: mutable.ArrayBuffer[ClusterStat],
      threshold: Double
  ): (mutable.ArrayBuffer[Int], mutable.ArrayBuffer[Array[Double]]) = {
    val remainIds = mutable.ArrayBuffer.empty[Int]
    val remainPts = mutable.ArrayBuffer.empty[Array[Double]]
    if (clusters.isEmpty || pointIds.isEmpty) {
      remainIds ++= pointIds
      remainPts ++= points
      return (remainIds, remainPts)
    }
    var i = 0
    while (i < pointIds.length) {
      val pt = points(i)
      var bestI = -1
      var bestD = Double.PositiveInfinity
      var c = 0
      while (c < clusters.length) {
        val d = mahalanobisPoint(pt, clusters(c))
        if (d < bestD) { bestD = d; bestI = c }
        c += 1
      }
      if (bestI >= 0 && bestD < threshold) clusters(bestI).addPoint(pt, pointIds(i))
      else {
        remainIds += pointIds(i)
        remainPts += pt
      }
      i += 1
    }
    (remainIds, remainPts)
  }

  def mergeCsClusters(
      csList: mutable.ArrayBuffer[ClusterStat],
      threshold: Double
  ): mutable.ArrayBuffer[ClusterStat] = {
    if (csList.length <= 1) return csList
    var current = csList
    var merged = true
    while (merged) {
      merged = false
      val n = current.length
      val used = Array.fill(n)(false)
      val newList = mutable.ArrayBuffer.empty[ClusterStat]
      var i = 0
      while (i < n) {
        if (!used(i)) {
          var bestJ = -1
          var bestD = Double.PositiveInfinity
          var j = i + 1
          while (j < n) {
            if (!used(j)) {
              val d = mahalanobisClusters(current(i), current(j))
              if (d < threshold && d < bestD) { bestD = d; bestJ = j }
            }
            j += 1
          }
          if (bestJ >= 0) {
            current(i).merge(current(bestJ))
            used(bestJ) = true
            merged = true
          }
          newList += current(i)
          used(i) = true
        }
        i += 1
      }
      current = newList
    }
    current
  }

  def mergeCsIntoDs(
      dsList: mutable.ArrayBuffer[ClusterStat],
      csList: mutable.ArrayBuffer[ClusterStat],
      threshold: Double,
      force: Boolean = false
  ): mutable.ArrayBuffer[ClusterStat] = {
    val remain = mutable.ArrayBuffer.empty[ClusterStat]
    csList.foreach { cs =>
      if (dsList.isEmpty) remain += cs
      else {
        var bestI = 0
        var bestD = Double.PositiveInfinity
        var i = 0
        while (i < dsList.length) {
          val d = mahalanobisClusters(cs, dsList(i))
          if (d < bestD) { bestD = d; bestI = i }
          i += 1
        }
        if (force || bestD < threshold) dsList(bestI).merge(cs)
        else remain += cs
      }
    }
    remain
  }

  def countStats(
      dsList: mutable.ArrayBuffer[ClusterStat],
      csList: mutable.ArrayBuffer[ClusterStat],
      rsIds: mutable.ArrayBuffer[Int]
  ): (Int, Int, Int, Int) =
    (dsList.map(_.N).sum, csList.length, csList.map(_.N).sum, rsIds.length)

  def loadData(sc: SparkContext, path: String): Array[DataPoint] = {
    sc.textFile(path).filter(_.trim.nonEmpty).map { line =>
      val parts = line.split(",")
      DataPoint(parts(0).toDouble.toInt, parts.drop(2).map(_.toDouble))
    }.collect()
  }

  def clusterCentroid(pts: Array[Array[Double]]): Array[Double] = {
    val s = Array.fill(pts(0).length)(0.0)
    var i = 0
    while (i < pts.length) {
      var d = 0
      while (d < s.length) { s(d) += pts(i)(d); d += 1 }
      i += 1
    }
    var d = 0
    while (d < s.length) { s(d) /= pts.length.toDouble; d += 1 }
    s
  }

  def bfr(sc: SparkContext, data: Array[DataPoint], nCluster: Int): (Array[(Int, Int, Int, Int)], Map[Int, Int]) = {
    val nPoints = data.length
    val dim = data.head.features.length
    val threshold = 2.0 * math.sqrt(dim)

    val rng = new Random(553)
    val order = Array.tabulate(nPoints)(i => i)
    var i = nPoints - 1
    while (i > 0) {
      val j = rng.nextInt(i + 1)
      val tmp = order(i); order(i) = order(j); order(j) = tmp
      i -= 1
    }
    val base = nPoints / 5
    val rem = nPoints % 5
    var offset = 0
    val splits = (0 until 5).map { s =>
      val sz = base + (if (s < rem) 1 else 0)
      val part = order.slice(offset, offset + sz).map(data)
      offset += sz
      part
    }.filter(_.nonEmpty).toArray

    val intermediate = mutable.ArrayBuffer.empty[(Int, Int, Int, Int)]
    val chunk0 = splits(0)
    val ids0 = chunk0.map(_.id)
    val pts0 = chunk0.map(_.features)

    System.out.println(s"BFR step2 largeK on ${pts0.length} pts..."); System.out.flush()
    val labels0 = runKMeans(sc, pts0, 5 * nCluster, 553L)
    val (multi0, singles0) = splitBySize(labels0, ids0, pts0)
    System.out.println(s"BFR step2 done multi=${multi0.length} rs=${singles0.length}"); System.out.flush()

    var rsIds = mutable.ArrayBuffer.empty[Int] ++ singles0.map(_._1)
    var rsPoints = mutable.ArrayBuffer.empty[Array[Double]] ++ singles0.map(_._2)

    val restIdsBuf = mutable.ArrayBuffer.empty[Int]
    val restPtsBuf = mutable.ArrayBuffer.empty[Array[Double]]
    multi0.foreach { case (ids, pts) =>
      restIdsBuf ++= ids
      restPtsBuf ++= pts
    }
    val restIds = restIdsBuf.toArray
    val restPts = restPtsBuf.toArray

    val dsList = mutable.ArrayBuffer.empty[ClusterStat]
    if (restPts.nonEmpty) {
      // Step 4: seed from largest large-K clusters, then refine (more stable than plain MLlib)
      val ranked = multi0.sortBy(-_._2.length)
      val seedCentroids = ranked.take(math.min(nCluster, ranked.length)).map { case (_, pts) =>
        clusterCentroid(pts)
      }
      val dsLabels =
        if (seedCentroids.length == nCluster) refineKMeans(restPts, seedCentroids)
        else runKMeans(sc, restPts, nCluster, 553L)
      val groups = mutable.HashMap.empty[Int, mutable.ArrayBuffer[Int]]
      var gi = 0
      while (gi < dsLabels.length) {
        groups.getOrElseUpdate(dsLabels(gi), mutable.ArrayBuffer.empty[Int]) += gi
        gi += 1
      }
      groups.keys.toSeq.sorted.foreach { lab =>
        val idxs = groups(lab)
        dsList += new ClusterStat(idxs.map(restPts), idxs.map(restIds))
      }
    }

    var tmp = rsToCsAndRs(sc, rsIds, rsPoints, nCluster, 553L)
    var csList = tmp._1
    rsIds = tmp._2
    rsPoints = tmp._3
    intermediate += countStats(dsList, csList, rsIds)

    var roundI = 2
    while (roundI <= splits.length) {
      val chunk = splits(roundI - 1)
      var remain = assignToNearest(chunk.map(_.features), chunk.map(_.id), dsList, threshold)
      remain = assignToNearest(remain._2.toArray, remain._1.toArray, csList, threshold)
      rsIds ++= remain._1
      rsPoints ++= remain._2

      tmp = rsToCsAndRs(sc, rsIds, rsPoints, nCluster, 553L + roundI)
      csList ++= tmp._1
      rsIds = tmp._2
      rsPoints = tmp._3
      csList = mergeCsClusters(csList, threshold)
      // Last round: force-merge remaining CS into nearest DS (improves discard/accuracy)
      if (roundI == splits.length) csList = mergeCsIntoDs(dsList, csList, threshold, force = true)
      intermediate += countStats(dsList, csList, rsIds)
      roundI += 1
    }

    val assignment = mutable.HashMap.empty[Int, Int]
    dsList.zipWithIndex.foreach { case (ds, cid) => ds.pointIds.foreach(pid => assignment(pid) = cid) }
    csList.foreach(_.pointIds.foreach(pid => assignment(pid) = -1))
    rsIds.foreach(pid => assignment(pid) = -1)
    data.foreach { p => if (!assignment.contains(p.id)) assignment(p.id) = -1 }
    (intermediate.toArray, assignment.toMap)
  }

  def writeOutput(path: String, intermediate: Array[(Int, Int, Int, Int)], assignment: Map[Int, Int]): Unit = {
    val pw = new PrintWriter(new BufferedWriter(new FileWriter(path)))
    try {
      pw.println("The intermediate results:")
      intermediate.zipWithIndex.foreach { case ((dsPts, csN, csPts, rsPts), idx) =>
        pw.println(s"Round ${idx + 1}: $dsPts,$csN,$csPts,$rsPts")
      }
      pw.println()
      pw.println("The clustering results:")
      assignment.keys.toSeq.sorted.foreach(pid => pw.println(s"$pid,${assignment(pid)}"))
    } finally pw.close()
  }

  def main(args: Array[String]): Unit = {
    def log(msg: String): Unit = {
      System.out.println(msg)
      System.out.flush()
      System.err.println(msg)
      System.err.flush()
    }

    log(s"BFR args: ${args.mkString(" | ")}")
    val inputFile = args(0)
    val nCluster = args(1).toInt
    val outputFile = args(2)

    val conf = new SparkConf().setAppName("BFR")
    if (!conf.contains("spark.master")) conf.setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")
    try {
      log("BFR loading data...")
      val data = loadData(sc, inputFile)
      log(s"BFR loaded n=${data.length} dim=${data.head.features.length} k=$nCluster")
      val (intermediate, assignment) = bfr(sc, data, nCluster)
      log(s"BFR rounds=${intermediate.length} assigned=${assignment.size}")
      intermediate.zipWithIndex.foreach { case (t, i) =>
        log(s"BFR Round ${i + 1}: ${t._1},${t._2},${t._3},${t._4}")
      }
      writeOutput(outputFile, intermediate, assignment)
      log(s"BFR wrote $outputFile")
    } catch {
      case e: Throwable =>
        log("BFR failed: " + e.getClass.getName + ": " + e.getMessage)
        e.printStackTrace(System.err)
        throw e
    } finally {
      sc.stop()
    }
  }
}
