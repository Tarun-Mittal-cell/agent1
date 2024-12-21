"""
End-to-end tests for authentication flow
"""
describe('Authentication Flow', () => {
  beforeEach(() => {
    cy.intercept('POST', '/api/v1/auth/login').as('loginRequest')
    cy.intercept('POST', '/api/v1/auth/register').as('registerRequest')
  })

  it('should successfully register a new user', () => {
    // Visit register page
    cy.visit('/register')

    // Fill registration form
    cy.get('[data-test="register-name"]').type('Test User')
    cy.get('[data-test="register-email"]').type('test@example.com')
    cy.get('[data-test="register-password"]').type('password123')
    cy.get('[data-test="register-confirm"]').type('password123')

    // Submit form
    cy.get('[data-test="register-submit"]').click()

    // Wait for request and verify redirect
    cy.wait('@registerRequest')
    cy.url().should('include', '/feed')

    // Verify user is logged in
    cy.get('[data-test="user-menu"]').should('exist')
  })

  it('should successfully login an existing user', () => {
    // Visit login page
    cy.visit('/login')

    // Fill login form
    cy.get('[data-test="login-email"]').type('test@example.com')
    cy.get('[data-test="login-password"]').type('password123')

    // Submit form
    cy.get('[data-test="login-submit"]').click()

    // Wait for request and verify redirect
    cy.wait('@loginRequest')
    cy.url().should('include', '/feed')

    // Verify user is logged in
    cy.get('[data-test="user-menu"]').should('exist')
  })

  it('should show error message for invalid credentials', () => {
    // Visit login page
    cy.visit('/login')

    // Fill login form with invalid credentials
    cy.get('[data-test="login-email"]').type('wrong@example.com')
    cy.get('[data-test="login-password"]').type('wrongpass')

    // Submit form
    cy.get('[data-test="login-submit"]').click()

    // Verify error message
    cy.get('[data-test="login-error"]')
      .should('be.visible')
      .and('contain', 'Invalid email or password')
  })

  it('should successfully logout', () => {
    // Login first
    cy.login('test@example.com', 'password123')

    // Click logout button
    cy.get('[data-test="user-menu"]').click()
    cy.get('[data-test="logout-button"]').click()

    // Verify redirect to login page
    cy.url().should('include', '/login')

    // Verify user is logged out
    cy.get('[data-test="user-menu"]').should('not.exist')
  })

  it('should maintain authentication state after refresh', () => {
    // Login
    cy.login('test@example.com', 'password123')

    // Refresh page
    cy.reload()

    // Verify still logged in
    cy.get('[data-test="user-menu"]').should('exist')
  })
})