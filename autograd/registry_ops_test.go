package autograd

import "testing"

// Every elementwise operation must also be registered for device dispatch, or
// dispatchElementwise panics at the first call.
func TestElementwiseRegistryMatchesOperationRegistry(t *testing.T) {
	for name := range elementwiseRegistry {
		if _, ok := operationRegistry[name]; !ok {
			t.Fatalf("%q is in elementwiseRegistry but not operationRegistry", name)
		}
	}
}
