package org.apache.spark.util;

import org.apache.spark.SparkEnv;
import org.apache.spark.SparkException;

/**
 * Experimental driver-only replacement for Spark 4.0.2 SparkClosureCleaner$: identical except that the
 * eager "is this closure serializable" check (a full Java serialization of every mapPartitions closure,
 * which for GPU plans re-serializes large captured expression trees) is skipped. Closures are still
 * cleaned, and are still serialized for real by DAGScheduler at task submission, where a
 * non-serializable closure still fails the job with "Task not serializable".
 */
public final class SparkClosureCleaner$ {
  public static final SparkClosureCleaner$ MODULE$ = new SparkClosureCleaner$();
  static final boolean CHECK = Boolean.getBoolean("spark.experimental.closureSerializableCheck");

  private SparkClosureCleaner$() {}

  public void clean(Object closure, boolean checkSerializable, boolean cleanTransitively) {
    if (ClosureCleaner$.MODULE$.clean(closure, cleanTransitively,
        (scala.collection.mutable.Map) scala.collection.mutable.Map$.MODULE$.empty())) {
      try {
        if (CHECK && checkSerializable && SparkEnv.get() != null) {
          SparkEnv.get().closureSerializer().newInstance()
              .serialize(closure, scala.reflect.ClassTag$.MODULE$.AnyRef());
        }
      } catch (Exception e) {
        throw SparkClosureCleaner$.<RuntimeException>sneaky(new SparkException("Task not serializable", e));
      }
    }
  }

  @SuppressWarnings("unchecked")
  private static <T extends Throwable> T sneaky(Throwable t) throws T { throw (T) t; }

  public boolean clean$default$2() { return true; }
  public boolean clean$default$3() { return true; }
}
