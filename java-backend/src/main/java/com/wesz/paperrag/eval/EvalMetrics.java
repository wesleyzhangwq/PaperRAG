package com.wesz.paperrag.eval;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

public record EvalMetrics(double recallAtK, double mrr, double ndcgAtK) {

    public static EvalMetrics calculate(List<Long> groundTruthIds, List<Long> retrievedIds, int k) {
        Set<Long> relevant = new HashSet<>(groundTruthIds == null ? List.of() : groundTruthIds);
        List<Long> retrieved = retrievedIds == null ? List.of() : retrievedIds;
        int boundedK = Math.max(1, k);
        if (relevant.isEmpty()) {
            return new EvalMetrics(0.0, 0.0, 0.0);
        }
        return new EvalMetrics(
            round4(recallAtK(relevant, retrieved, boundedK)),
            round4(mrr(relevant, retrieved)),
            round4(ndcgAtK(relevant, retrieved, boundedK))
        );
    }

    private static double recallAtK(Set<Long> relevant, List<Long> retrieved, int k) {
        long hits = retrieved.stream()
            .limit(k)
            .filter(relevant::contains)
            .count();
        return hits / (double) relevant.size();
    }

    private static double mrr(Set<Long> relevant, List<Long> retrieved) {
        for (int index = 0; index < retrieved.size(); index++) {
            if (relevant.contains(retrieved.get(index))) {
                return 1.0 / (index + 1);
            }
        }
        return 0.0;
    }

    private static double ndcgAtK(Set<Long> relevant, List<Long> retrieved, int k) {
        double dcg = 0.0;
        for (int index = 0; index < Math.min(k, retrieved.size()); index++) {
            if (relevant.contains(retrieved.get(index))) {
                dcg += gain(index);
            }
        }
        int idealHits = Math.min(k, relevant.size());
        double idcg = 0.0;
        for (int index = 0; index < idealHits; index++) {
            idcg += gain(index);
        }
        return idcg == 0.0 ? 0.0 : dcg / idcg;
    }

    private static double gain(int zeroBasedRank) {
        return 1.0 / (Math.log(zeroBasedRank + 2) / Math.log(2));
    }

    private static double round4(double value) {
        return Math.round(value * 10_000.0) / 10_000.0;
    }
}
