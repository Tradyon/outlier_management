UPDATE `tradyon.shipment_master_beta_20250728` t
SET
  is_outlier = TRUE
FROM (
  SELECT
    SUBSTR(hs_code, 1, 6) AS hs_code,
    APPROX_QUANTILES(unit_price, 2)[OFFSET(1)] AS median_unit_price
  FROM `tradyon.shipment_master_beta_20250728`
  WHERE unit_price IS NOT NULL
    AND is_multi_product_shipment = FALSE
  GROUP BY SUBSTR(hs_code, 1, 6)
) m
WHERE
  SUBSTR(t.hs_code, 1, 6) = m.hs_code
  AND t.is_multi_product_shipment = FALSE
  AND m.median_unit_price > 0
  AND (
       t.unit_price > 20 * m.median_unit_price
    OR t.unit_price < 0.05 * m.median_unit_price
  );
