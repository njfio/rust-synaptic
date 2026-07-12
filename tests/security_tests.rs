//! Comprehensive tests for the security module

#![cfg(feature = "security")]

use synaptic::{
    error::Result,
    security::access_control::{
        AccessControlManager, AuthenticationCredentials, AuthenticationType,
    },
    security::{Permission, SecurityConfig, SecurityManager},
};

/// Provision real credentials (argon2 password + TOTP secret) for `user_id`
/// and return credentials carrying a valid current TOTP token (Task 4.7:
/// authentication is real now, so users must be provisioned first).
fn provision_and_credentials(
    access_control: &mut AccessControlManager,
    user_id: &str,
    password: &str,
) -> Result<AuthenticationCredentials> {
    let secret = AccessControlManager::generate_totp_secret();
    access_control.set_password(user_id, password)?;
    access_control.set_totp_secret(user_id, secret.clone())?;
    let totp = totp_rs::TOTP::new(totp_rs::Algorithm::SHA1, 6, 1, 30, secret)
        .expect("generated TOTP secret is valid");
    let token = totp.generate_current().expect("system clock is available");
    Ok(AuthenticationCredentials {
        auth_type: AuthenticationType::Password,
        password: Some(password.to_string()),
        api_key: None,
        certificate: None,
        mfa_token: Some(token),
        ip_address: Some("127.0.0.1".to_string()),
        user_agent: Some("test".to_string()),
    })
}

#[tokio::test]
async fn test_security_manager_creation() -> Result<()> {
    let config = SecurityConfig::default();
    let _manager = SecurityManager::new(config).await?;
    Ok(())
}

#[tokio::test]
async fn test_access_control_authentication() -> Result<()> {
    let config = SecurityConfig::default();
    let mut access_control = AccessControlManager::new(&config).await?;

    // Add a role with permissions
    access_control
        .add_role(
            "user".to_string(),
            vec![Permission::ReadMemory, Permission::WriteMemory],
        )
        .await?;

    // Test authentication with valid credentials
    let creds = provision_and_credentials(&mut access_control, "user", "password123")?;

    let context = access_control
        .authenticate("user".to_string(), creds)
        .await?;
    assert_eq!(context.user_id, "user");
    assert!(!context.roles.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_permission_checking() -> Result<()> {
    let config = SecurityConfig::default();
    let mut access_control = AccessControlManager::new(&config).await?;

    // Add roles with different permissions
    access_control
        .add_role(
            "admin".to_string(),
            vec![
                Permission::ReadMemory,
                Permission::WriteMemory,
                Permission::DeleteMemory,
            ],
        )
        .await?;

    access_control
        .add_role(
            "user".to_string(),
            vec![Permission::ReadMemory, Permission::WriteMemory],
        )
        .await?;

    // Test admin authentication
    let admin_creds = provision_and_credentials(&mut access_control, "admin", "admin_password")?;

    let admin_ctx = access_control
        .authenticate("admin".to_string(), admin_creds)
        .await?;

    // Admin should have delete permission
    assert!(access_control
        .check_permission(&admin_ctx, Permission::DeleteMemory)
        .await
        .is_ok());

    // Test user authentication
    let user_creds = provision_and_credentials(&mut access_control, "user", "user_password")?;

    let user_ctx = access_control
        .authenticate("user".to_string(), user_creds)
        .await?;

    // User should have read permission
    assert!(access_control
        .check_permission(&user_ctx, Permission::ReadMemory)
        .await
        .is_ok());

    // User should NOT have delete permission
    assert!(access_control
        .check_permission(&user_ctx, Permission::DeleteMemory)
        .await
        .is_err());

    Ok(())
}

#[tokio::test]
async fn test_session_management() -> Result<()> {
    let config = SecurityConfig::default();
    let mut access_control = AccessControlManager::new(&config).await?;

    // Add a role
    access_control
        .add_role("user".to_string(), vec![Permission::ReadMemory])
        .await?;

    // Authenticate and create session
    let creds = provision_and_credentials(&mut access_control, "user", "password123")?;

    let context = access_control
        .authenticate("user".to_string(), creds)
        .await?;

    // Session should be valid
    assert!(access_control.validate_session(&context).await.is_ok());

    // Revoke session
    access_control.revoke_session(&context.session_id).await?;

    // Session should now be invalid
    assert!(access_control.validate_session(&context).await.is_err());

    Ok(())
}

#[tokio::test]
async fn test_access_control_metrics() -> Result<()> {
    let config = SecurityConfig::default();
    let mut access_control = AccessControlManager::new(&config).await?;

    // Add a role
    access_control
        .add_role("user".to_string(), vec![Permission::ReadMemory])
        .await?;

    // Perform some operations
    let creds = provision_and_credentials(&mut access_control, "user", "password123")?;

    let context = access_control
        .authenticate("user".to_string(), creds)
        .await?;
    let _ = access_control
        .check_permission(&context, Permission::ReadMemory)
        .await;

    // Get metrics
    let metrics = access_control.get_metrics().await?;
    assert!(metrics.total_successful_authentications > 0);
    assert!(metrics.total_permission_checks > 0);

    Ok(())
}
