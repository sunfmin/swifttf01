import XCTest

#if !os(macOS)
public func allTests() -> [XCTestCaseEntry] {
    return [
        testCase(swifttf01Tests.allTests),
    ]
}
#endif
