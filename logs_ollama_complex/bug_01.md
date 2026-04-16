# Coupon applies incorrect discount

**Severity:** Medium

## Expected
SUMMER20 should reduce a $100 subtotal by $20.

## Actual
The order summary shows Discount -$10 after applying SUMMER20.

## Reason
The pricing logic appears to apply the wrong discount amount.

## Steps to Reproduce
1. Typed into "Search products".
2. Selected "Outerwear" for "Category".
3. Clicked 'Apply Filters'.
4. Clicked 'View Product'.
5. Clicked 'Open Size Guide'.
6. Clicked 'Close Size Guide'.
7. Selected "M" for "Size".
8. Clicked 'Add to Cart'.
9. Clicked 'Cart (1)'.
10. Typed into "Coupon".
11. Clicked 'Apply Coupon'.
12. Clicked 'Estimate Shipping'.
